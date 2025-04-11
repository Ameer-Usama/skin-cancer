import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0083B8;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0083B8;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        line-height: 1.6;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #888;
        font-size: 0.8rem;
    }
    .stButton>button {
        background-color: #0083B8;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #006491;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model/skin_cancer_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to make prediction
def predict_skin_cancer(model, img_array):
    # Class names (update these based on your model's classes)
    class_names = [
        "Actinic Keratoses",
        "Basal Cell Carcinoma",
        "Benign Keratosis",
        "Dermatofibroma",
        "Melanoma",
        "Melanocytic Nevi",
        "Vascular Lesions"
    ]
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    return {
        "class": class_names[predicted_class],
        "confidence": confidence,
        "all_probabilities": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }

# Function to display results
def display_results(result, uploaded_image):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown(f"<h2 class='sub-header'>Diagnosis Results</h2>", unsafe_allow_html=True)
        
        # Determine result color based on confidence
        if result["confidence"] > 0.7:
            result_color = "#FF4B4B" if result["class"] == "Melanoma" else "#00A36C"
        else:
            result_color = "#FFA500"  # Orange for uncertain results
        
        # Display main result
        st.markdown(f"""
        <div class='result-box' style='background-color: {result_color}20; border-left: 5px solid {result_color};'>
            <h3 style='color: {result_color};'>Detected: {result["class"]}</h3>
            <p>Confidence: {result["confidence