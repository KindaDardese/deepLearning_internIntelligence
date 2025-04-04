import os
import streamlit as st
import numpy as np
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import pickle
from PIL import Image
from keras._tf_keras.keras.applications import DenseNet201

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@st.cache_resource
def load_models():
    """Load the captioning model and tokenizer"""
    try:
        # Load caption generation model
        caption_model = load_model("output_results/best_model.keras")
        
        # Initialize feature extractor
        feature_extractor = DenseNet201(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        
        # Load tokenizer
        with open("output_results/tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)
            
        return caption_model, feature_extractor, tokenizer
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.error("Please ensure you have:")
        st.error("1. best_model.keras")
        st.error("2. tokenizer.pickle")
        st.error("in the output_results folder")
        return None, None, None

def generate_caption(image, max_length=34):
    """Generate caption for uploaded image"""
    caption_model, feature_extractor, tokenizer = load_models()
    
    if None in [caption_model, feature_extractor, tokenizer]:
        return "Model loading failed - please check error messages above"
    
    try:
        # Preprocess image
        img = Image.open(image)
        img = img.resize((224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        features = feature_extractor.predict(img, verbose=0)
        
        # Generate caption
        caption = 'startseq'
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([caption])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            
            yhat = caption_model.predict([features, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat, None)
            
            if word is None or word == "endseq":
                break
                
            caption += ' ' + word
        
        return caption.replace('startseq ', '').replace(' endseq', '')
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Image Caption Generator",
        layout="centered",
        page_icon="🖼️"
    )
    
    st.title("🖼️ AI Image Caption Generator")
    st.markdown("""
    Upload an image to generate an automatic description using our deep learning model.
    The system uses a CNN-LSTM architecture trained on thousands of images.
    """)
    
    with st.sidebar:
        st.header("Settings")
        max_length = st.slider(
            "Maximum caption length", 
            min_value=10,
            max_value=50,
            value=34,
            help="Longer captions may be less accurate"
        )
        
        st.markdown("---")
        st.markdown("**Model Information**")
        st.markdown("- **Backbone**: DenseNet201")
        st.markdown("- **Decoder**: LSTM with Attention")
        st.markdown("- **Input Size**: 224×224 pixels")
    
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1.5, 1], gap="large")
        
        with col1:
            st.subheader("Your Image")
            st.image(
                uploaded_file,
                caption="Uploaded image",
                use_container_width=True
            )
            
            if st.button("✨ Generate Caption", type="primary", use_container_width=True):
                with st.spinner("Generating caption..."):
                    caption = generate_caption(uploaded_file, max_length)
                
                st.subheader("Generated Caption:")
                st.success(caption)
                
                # Store in session state
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append((uploaded_file.name, caption))
        
        with col2:
            st.subheader("How It Works")
            st.markdown("""
            1. **Feature Extraction**:
               - Image processed through DenseNet201
               - 2048-dimensional feature vector extracted
            
            2. **Caption Generation**:
               - LSTM decoder generates words sequentially
               - Attention mechanism focuses on relevant image parts
               - Stops at "endseq" token or max length
            """)
            
            if 'history' in st.session_state and st.session_state.history:
                st.markdown("---")
                st.subheader("Recent Captions")
                for name, cap in st.session_state.history[-3:]:
                    st.caption(f"**{name}**")
                    st.code(cap, language="text")

if __name__ == "__main__":
    main()