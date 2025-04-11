import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import io
import os
from utils import preprocess_image, get_gradcam, get_image_download_link, plot_prediction_bars

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
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0083B8;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #006491;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .tabs {
        border-radius: 10px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #0083B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0083B8 !important;
        color: white !important;
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
        "all_probabilities": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))},
        "is_malignant": predicted_class in [1, 4]  # Basal Cell Carcinoma and Melanoma are malignant
    }

# Function to display results
def display_results(result, uploaded_image, gradcam_image=None):
    # Create tabs for different views
    tabs = st.tabs(["Analysis Results", "Detailed View", "Technical Information"])
    
    # Tab 1: Main Results
    with tabs[0]:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Add download link for the image
            st.markdown(get_image_download_link(uploaded_image, "original_image.jpg", "Download Original Image"), unsafe_allow_html=True)
            
            if gradcam_image is not None:
                st.image(gradcam_image, caption="GradCAM Visualization", use_column_width=True)
                st.markdown(get_image_download_link(gradcam_image, "gradcam_image.jpg", "Download GradCAM Image"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<h2 class='sub-header'>Diagnosis Results</h2>", unsafe_allow_html=True)
            
            # Determine result color based on malignancy
            result_color = "#FF4B4B" if result["is_malignant"] else "#00A36C"
            result_text = "Potentially Malignant" if result["is_malignant"] else "Likely Benign"
            
            # Display main result
            st.markdown(f"""
            <div class='result-box' style='background-color: {result_color}20; border-left: 5px solid {result_color};'>
                <h3 style='color: {result_color};'>Detected: {result["class"]}</h3>
                <h4 style='color: {result_color};'>Assessment: {result_text}</h4>
                <p>Confidence: {result["confidence