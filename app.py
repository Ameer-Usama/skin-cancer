import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load your trained model
try:
    model = load_model('skin_cancer_model.h5')  # Ensure model path is correct
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Class names (update with your actual classes)
class_names = ['Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma', 
              'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion']

# Configure page
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f9ff;
    }
    .uploadedImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🔍 Skin Cancer Detection Analysis")
st.markdown("Upload a clear image of skin lesion for AI-powered analysis")

# Image upload section
col1, col2 = st.columns([2, 3])
with col1:
    uploaded_file = st.file_uploader("Choose skin lesion image", 
                                   type=["jpg", "jpeg", "png"],
                                   help="Upload a clear photo of the skin lesion")

# Processing and results
if uploaded_file is not None:
    try:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True, output_format='auto', clamp=True)

        # Preprocess image
        img = np.array(image)
        img = cv2.resize(img, (300, 300))  # Updated to 300x300
        img = img / 255.0  # Normalization
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Prediction
        with col2:
            if st.button('Analyze Now', type='primary', use_container_width=True):
                with st.spinner('AI Analysis in Progress...'):
                    predictions = model.predict(img)
                    predicted_class = np.argmax(predictions)
                    confidence = np.max(predictions) * 100

                st.success('Analysis Complete')
                st.subheader("Diagnostic Results")
                
                # Main prediction
                st.metric(label="Most Likely Diagnosis", 
                         value=f"{class_names[predicted_class]}", 
                         delta=f"{confidence:.2f}% confidence")
                
                # Detailed probabilities
                with st.expander("View Detailed Probabilities"):
                    probs = predictions[0]
                    for i, (class_name, prob) in enumerate(zip(class_names, probs)):
                        st.progress(prob, text=f"{class_name}: {prob*100:.2f}%")

                # Medical disclaimer
                st.warning("""
                **Important Note:**  
                This AI analysis is a preliminary assessment and should NOT be considered as medical diagnosis. 
                Always consult a qualified dermatologist for professional medical evaluation.
                """)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Sidebar information
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This AI-powered diagnostic tool analyzes skin lesions using a deep learning model 
    trained on dermatoscopic images. Key features:
    
    - 300x300 image analysis resolution
    - 7-class skin cancer detection
    - Probability distribution visualization
    - Instant results delivery
    """)
    
    st.divider()
    st.markdown("**Supported Image Formats:**")
    st.markdown("- JPEG/JPG\n- PNG\n- High-quality images only")
    st.markdown("**Recommended Practices:**")
    st.markdown("- Use good lighting\n- Capture lesion center\n- Avoid hair obstruction")

# Footer
st.markdown("---")
st.markdown("🩺 Medical AI System | v2.1 | © 2024 SkinCare Diagnostics")