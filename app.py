import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("cats_vs_dogs_cnn.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, image):
    prediction = model.predict(image, verbose=0)
    probability = prediction[0][0]
    
    if probability > 0.5:
        return "Dog", probability
    else:
        return "Cat", 1 - probability

def main():
    st.markdown('<div class="main-header">🐱 Cat vs Dog Classifier 🐶</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an image and let AI determine if it\'s a cat or a dog!</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a Convolutional Neural Network (CNN) trained on thousands of cat and dog images 
        to classify your uploaded images.
        """)
        
        st.header("📊 Model Information")
        st.write("""
        - **Architecture**: Sequential CNN
        - **Input Size**: 224x224 pixels
        - **Training Dataset**: PetImages Dataset
        - **Framework**: TensorFlow/Keras
        """)
        
        st.header("🚀 How to Use")
        st.write("""
        1. Upload an image using the file uploader
        2. Wait for the model to process
        3. View the prediction and confidence score
        """)
        
        st.header("⚙️ Settings")
        show_confidence = st.checkbox("Show confidence score", value=True)
        show_original = st.checkbox("Show original image", value=True)
    
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG image of a cat or dog"
    )
    
    if uploaded_file is not None:
        model = load_model()
        
        if model is None:
            st.error("⚠️ Failed to load the model. Please check if 'cats_vs_dogs_cnn.keras' exists in the project directory.")
            return
        
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if show_original:
                st.subheader("📸 Original Image")
                st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🔮 Prediction")
            
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(image)
                animal, confidence = predict(model, processed_image)
                
                emoji = "🐶" if animal == "Dog" else "🐱"
                st.markdown(
                    f'<div class="prediction-box">{emoji} {animal}!</div>',
                    unsafe_allow_html=True
                )
                
                if show_confidence:
                    confidence_percent = confidence * 100
                    st.markdown(
                        f'<div class="confidence-box">Confidence: {confidence_percent:.2f}%</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.progress(confidence)
                
                st.success("✅ Classification complete!")
        
        st.markdown("---")
        result_text = f"Prediction: {animal}\nConfidence: {confidence*100:.2f}%"
        st.download_button(
            label="📥 Download Result",
            data=result_text,
            file_name="prediction_result.txt",
            mime="text/plain"
        )
    
    else:
        st.info("👆 Please upload an image to get started!")
        
        st.markdown("---")
        st.subheader("💡 Tips for Best Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("✅ **Good Quality Images**")
            st.write("Use clear, well-lit photos")
        
        with col2:
            st.write("🎯 **Single Animal**")
            st.write("Image should contain one cat or dog")
        
        with col3:
            st.write("📐 **Any Size Welcome**")
            st.write("Images will be automatically resized")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p>Built with ❤️ using Streamlit and TensorFlow</p>
            <p>Model trained on PetImages dataset</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
