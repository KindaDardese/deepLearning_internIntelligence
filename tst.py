import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import streamlit as st
import numpy as np
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle

caption_model = None
feature_extractor = None
tokenizer = None

def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):
    """
    Generate and display caption for an image using pre-trained models
    
    Args:
        image_path (str): Path to input image
        model_path (str): Path to caption model
        tokenizer_path (str): Path to tokenizer pickle file
        feature_extractor_path (str): Path to feature extractor model
        max_length (int): Maximum caption length
        img_size (int): Image dimensions for processing
    """
    global caption_model, feature_extractor, tokenizer
    
    # Load models only once and cache them
    if caption_model is None:        
        with st.spinner('Loading models...'):
            caption_model = load_model(model_path)
            feature_extractor = load_model(feature_extractor_path)
            
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)

    # Process image
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract image features
    with st.spinner('Extracting image features...'):
        image_features = feature_extractor.predict(img_array, verbose=0) 

    # Generate caption word by word
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        
        # Stop if word not found or end token predicted
        if word is None or word == "endseq":
            break
            
        in_text += " " + word

    # Clean up the generated caption
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    # Display results
    st.image(img, caption=f"Generated Caption: {caption}", use_column_width=True)
    
    with st.expander("Show prediction details"):
        st.write(f"Final caption: {caption}")
        st.write(f"Processing steps: {max_length}")
        st.write(f"Image size: {img_size}x{img_size}")

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Image Caption Generator", layout="wide")
    
    st.title("🖼️ Image Caption Generator")
    st.markdown("""
    Upload an image and generate descriptive captions using deep learning model.
    The system uses a CNN-LSTM architecture trained on thousands of image-caption pairs.
    """)
    
    # Sidebar for advanced options
    with st.sidebar:
        st.header("Settings")
        max_length = st.slider("Max caption length", 10, 50, 34)
        img_size = st.selectbox("Image size", [224, 299, 384], index=0)
        show_advanced = st.checkbox("Show advanced options")
        
        if show_advanced:
            model_path = st.text_input("Model path", "model.keras")
            tokenizer_path = st.text_input("Tokenizer path", "tokenizer.pkl")
            feature_path = st.text_input("Feature extractor path", "feature_extractor.keras")
        else:
            model_path = "model.keras"
            tokenizer_path = "tokenizer.pkl"
            feature_path = "feature_extractor.keras"
    
    # Main file uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_image_path = "temp_upload.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Generate and display caption
        try:
            generate_and_display_caption(
                temp_image_path, 
                model_path, 
                tokenizer_path, 
                feature_path,
                max_length=max_length,
                img_size=img_size
            )
        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")
            st.info("Please check that all model files are in the correct paths")
        
        os.remove(temp_image_path)

if __name__ == "__main__":
    main()