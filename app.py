import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Set page config
st.set_page_config(
    page_title="Skin Cancer Detection System",
    page_icon="🩺",
    layout="wide"
)

# Load your model (update the path to your model)
@st.cache_resource
def load_cancer_model():
    try:
        model = load_model('skin_cancer_model.h5')  # Update with your model path
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cancer_model()

# Define class labels (update these based on your model's classes)
CLASS_LABELS = {
    0: "Actinic Keratosis",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic Nevus",
    6: "Vascular Lesion"
}

# Preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert RGBA to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 3:
            pass  # Already RGB
        
        # Resize and normalize
        image = cv2.resize(image, target_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Main function
def main():
    st.title("🩺 Skin Cancer Detection System")
    st.write("Upload an image of a skin lesion to get a prediction.")
    
    # Sidebar with information
    st.sidebar.title("About")
    st.sidebar.info(
        "This system uses deep learning to analyze skin lesions and provide potential diagnoses. "
        "It is intended for educational purposes only and not as a substitute for professional medical advice."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.write("1. Upload a clear image of the skin lesion")
    st.sidebar.write("2. Ensure the lesion is centered and visible")
    st.sidebar.write("3. Click 'Analyze' to get the prediction")
    
    st.sidebar.title("Disclaimer")
    st.sidebar.warning(
        "This tool is not a substitute for professional medical diagnosis. "
        "Always consult a dermatologist for any concerning skin lesions."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analyze button
            if st.button("Analyze Image"):
                with st.spinner("Analyzing the image..."):
                    try:
                        # Preprocess the image
                        processed_image = preprocess_image(image)
                        
                        if processed_image is not None and model is not None:
                            # Make prediction
                            predictions = model.predict(processed_image)
                            predicted_class = np.argmax(predictions[0])
                            confidence = np.max(predictions[0]) * 100
                            
                            # Display results
                            st.success("Analysis Complete!")
                            
                            with col2:
                                st.subheader("Prediction Results")
                                st.write(f"**Predicted Condition:** {CLASS_LABELS.get(predicted_class, 'Unknown')}")
                                st.write(f"**Confidence:** {confidence:.2f}%")
                                
                                # Show prediction probabilities
                                st.subheader("Prediction Probabilities")
                                for i, prob in enumerate(predictions[0]):
                                    st.progress(float(prob), text=f"{CLASS_LABELS.get(i, f'Class {i}')}: {prob*100:.2f}%")
                                
                                # Warning for serious conditions
                                if predicted_class in [0, 1, 4]:  # High-risk conditions
                                    st.error("⚠️ This prediction suggests a potentially serious condition. Please consult a dermatologist immediately.")
                                elif predicted_class in [2, 3, 5, 6]:  # Lower-risk conditions
                                    st.info("ℹ️ This prediction suggests a less serious condition, but you should still consult a doctor if concerned.")
                                
                                # Disclaimer
                                st.warning("Remember: This tool is not a substitute for professional medical advice.")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()