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
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'skin_cancer_model.h5')
        model = tf.keras.models.load_model(model_path)
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
                <p>Confidence: {result["confidence"]:0.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display probability bars
            st.markdown("<h3>Probability Distribution</h3>", unsafe_allow_html=True)
            fig = plot_prediction_bars(result["all_probabilities"], list(result["all_probabilities"].keys()))
            st.pyplot(fig)
    
    # Tab 2: Detailed View
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Detailed Analysis</h2>", unsafe_allow_html=True)
        
        # Information about the detected condition
        condition_info = {
            "Actinic Keratoses": {
                "description": "Rough, scaly patches on the skin caused by years of sun exposure.",
                "risk_level": "Pre-cancerous",
                "common_locations": "Face, lips, ears, back of hands, forearms, scalp",
                "treatment": "Cryotherapy, topical medications, photodynamic therapy"
            },
            "Basal Cell Carcinoma": {
                "description": "The most common type of skin cancer, appears as a pearly or waxy bump.",
                "risk_level": "Malignant, but rarely spreads",
                "common_locations": "Face, neck, and other sun-exposed areas",
                "treatment": "Surgical removal, radiation therapy, topical medications"
            },
            "Benign Keratosis": {
                "description": "Harmless growths that appear as waxy, scaly, slightly raised bumps.",
                "risk_level": "Benign (non-cancerous)",
                "common_locations": "Face, chest, shoulders, back",
                "treatment": "Usually no treatment needed; can be removed for cosmetic reasons"
            },
            "Dermatofibroma": {
                "description": "Small, firm bumps that usually appear on the legs.",
                "risk_level": "Benign (non-cancerous)",
                "common_locations": "Legs, arms, trunk",
                "treatment": "Usually no treatment needed; surgical removal if desired"
            },
            "Melanoma": {
                "description": "The most serious form of skin cancer, develops in melanocytes (cells that produce melanin).",
                "risk_level": "Highly malignant, can spread rapidly",
                "common_locations": "Can occur anywhere, often on the back for men and legs for women",
                "treatment": "Surgery, immunotherapy, targeted therapy, radiation, chemotherapy"
            },
            "Melanocytic Nevi": {
                "description": "Common moles, usually brown or black, can appear anywhere on the skin.",
                "risk_level": "Benign (non-cancerous), but should be monitored for changes",
                "common_locations": "Can appear anywhere on the body",
                "treatment": "No treatment needed unless suspicious changes occur"
            },
            "Vascular Lesions": {
                "description": "Abnormalities of blood vessels, including hemangiomas and port-wine stains.",
                "risk_level": "Usually benign",
                "common_locations": "Can appear anywhere on the body",
                "treatment": "Laser therapy, sclerotherapy, surgical removal"
            }
        }
        
        detected_condition = result["class"]
        info = condition_info.get(detected_condition, {})
        
        if info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""<div class='card'>
                    <h3>{detected_condition}</h3>
                    <p class='info-text'>{info.get('description', 'No description available.')}</p>
                    <p><strong>Risk Level:</strong> {info.get('risk_level', 'Unknown')}</p>
                </div>""", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""<div class='card'>
                    <h3>Clinical Information</h3>
                    <p><strong>Common Locations:</strong> {info.get('common_locations', 'Various')}</p>
                    <p><strong>Typical Treatments:</strong> {info.get('treatment', 'Consult a dermatologist')}</p>
                </div>""", unsafe_allow_html=True)
        
        # General advice
        st.markdown("""<div class='card'>
            <h3>General Advice</h3>
            <p class='info-text'>
                <strong>Remember:</strong> This is an AI-based analysis and should not replace professional medical advice.
                If you have concerns about a skin lesion, please consult a dermatologist or healthcare provider.
                <br><br>
                Regular skin self-examinations and professional skin checks are recommended, especially if you have risk factors such as:
                <ul>
                    <li>Fair skin</li>
                    <li>History of sunburns</li>
                    <li>Excessive sun exposure</li>
                    <li>Family history of skin cancer</li>
                    <li>Personal history of skin cancer</li>
                    <li>Weakened immune system</li>
                </ul>
            </p>
        </div>""", unsafe_allow_html=True)
    
    # Tab 3: Technical Information
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Technical Information</h2>", unsafe_allow_html=True)
        
        # Model information
        st.markdown("""<div class='card'>
            <h3>About the Model</h3>
            <p class='info-text'>
                This application uses a deep learning model based on a convolutional neural network (CNN) architecture.
                The model was trained on thousands of dermatoscopic images of skin lesions across seven different categories.
                <br><br>
                <strong>Model Architecture:</strong> CNN with transfer learning
                <br>
                <strong>Input Size:</strong> 224x224 pixels
                <br>
                <strong>Classes:</strong> 7 (Actinic Keratoses, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanoma, Melanocytic Nevi, Vascular Lesions)
            </p>
        </div>""", unsafe_allow_html=True)
        
        # Raw probabilities
        st.markdown("<h3>Raw Prediction Data</h3>", unsafe_allow_html=True)
        st.json(result["all_probabilities"])
        
        # Explanation of GradCAM
        if gradcam_image is not None:
            st.markdown("""<div class='card'>
                <h3>About GradCAM Visualization</h3>
                <p class='info-text'>
                    Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique that produces a visual explanation of the regions
                    the neural network focused on when making its prediction. The heatmap highlights areas that influenced the model's decision,
                    with warmer colors (red) indicating regions of higher importance.
                </p>
            </div>""", unsafe_allow_html=True)

# Main application code
def main():
    # Display header
    st.markdown("<h1 class='main-header'>Skin Cancer Detection</h1>", unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the model file and try again.")
        return
    
    # Sidebar
    st.sidebar.markdown("<h2 class='sub-header'>Upload Image</h2>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
    
    # Sample images
    st.sidebar.markdown("<h3>Or try a sample image:</h3>", unsafe_allow_html=True)
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        
        if sample_images:
            sample_cols = st.sidebar.columns(min(3, len(sample_images)))
            selected_sample = None
            
            for i, col in enumerate(sample_cols):
                if i < len(sample_images):
                    img_path = os.path.join(sample_dir, sample_images[i])
                    img = Image.open(img_path)
                    col.image(img, width=80, caption=f"Sample {i+1}")
                    if col.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                        selected_sample = img_path
            
            if selected_sample:
                uploaded_file = selected_sample
    
    # Disclaimer
    st.sidebar.markdown("""<div class='footer'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
    </div>""", unsafe_allow_html=True)
    
    # Main content
    if uploaded_file is None:
        # Display information when no image is uploaded
        st.markdown("""<div class='card'>
            <h2 class='sub-header'>Welcome to the Skin Cancer Detection Tool</h2>
            <p class='info-text'>
                This application uses artificial intelligence to analyze images of skin lesions and identify potential skin conditions, including skin cancer.
                <br><br>
                To get started, upload an image of a skin lesion using the sidebar on the left.
                <br><br>
                <strong>Important:</strong> This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.
            </p>
        </div>""", unsafe_allow_html=True)
        
        # Display information about skin cancer
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""<div class='card'>
                <h3>About Skin Cancer</h3>
                <p class='info-text'>
                    Skin cancer is the abnormal growth of skin cells, most often developing on skin exposed to the sun. It can also occur on areas of your skin not ordinarily exposed to sunlight.
                    <br><br>
                    There are three major types of skin cancer:
                    <ul>
                        <li><strong>Basal cell carcinoma</strong></li>
                        <li><strong>Squamous cell carcinoma</strong></li>
                        <li><strong>Melanoma</strong></li>
                    </ul>
                </p>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""<div class='card'>
                <h3>Warning Signs</h3>
                <p class='info-text'>
                    Remember the ABCDE rule to identify potential melanoma:
                    <ul>
                        <li><strong>A</strong>symmetry: One half doesn't match the other</li>
                        <li><strong>B</strong>order: Irregular, ragged, notched, or blurred edges</li>
                        <li><strong>C</strong>olor: Different colors in the same mole</li>
                        <li><strong>D</strong>iameter: Larger than 6mm (pencil eraser)</li>
                        <li><strong>E</strong>volving: Changing in size, shape, or color</li>
                    </ul>
                </p>
            </div>""", unsafe_allow_html=True)
    
    else:
        # Process the uploaded image
        try:
            # Handle both uploaded file and sample image path
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file)
            else:
                image = Image.open(uploaded_file)
            
            # Preprocess the image
            img_array = preprocess_image(image)
            
            # Generate GradCAM visualization
            try:
                gradcam_image = get_gradcam(model, img_array)
                if gradcam_image is not None:
                    gradcam_image = Image.fromarray(gradcam_image)
            except Exception as e:
                st.warning(f"Could not generate GradCAM visualization: {e}")
                gradcam_image = None
            
            # Make prediction
            result = predict_skin_cancer(model, img_array)
            
            # Display results
            display_results(result, image, gradcam_image)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Run the application
if __name__ == "__main__":
    main()