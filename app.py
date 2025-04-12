import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load your trained model
model = load_model('skin_cancer_model.h5')  # Replace with your model path

# Class names (replace with your actual class names)
class_names = ['Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma', 
              'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion']

# Set page config
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺")

# Custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🩺 Skin Cancer Detection App")
st.markdown("Upload an image of skin lesion for analysis")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display image and process
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Adjust size according to your model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    if st.button('Analyze Image'):
        with st.spinner('Analyzing...'):
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            
        st.success('Analysis Complete!')
        st.subheader("Results")
        st.markdown(f"**Predicted Class:** {class_names[predicted_class]}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Show probabilities
        st.subheader("Class Probabilities")
        probs = predictions[0]
        for i, (class_name, prob) in enumerate(zip(class_names, probs)):
            st.markdown(f"- {class_name}: {prob*100:.2f}%")
            
        # Show disclaimer
        st.warning("**Disclaimer:** This analysis is not a substitute for professional medical advice. Always consult a healthcare professional.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info(
    """
    This AI-powered app helps in preliminary detection of skin cancer types using deep learning. 
    Upload an image of skin lesion to get analysis.
    
    **Note:** Results are not 100% accurate and should be verified by a dermatologist.
    """
)

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name] | © 2023 All rights reserved")