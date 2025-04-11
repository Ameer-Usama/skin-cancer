# Skin Cancer Detection Web Application

This is a Streamlit web application for skin cancer detection using a deep learning model. Users can upload skin lesion images, and the system will analyze them to detect potential skin cancer.

## Features

- User-friendly interface for image upload
- Real-time analysis of skin lesion images
- Detailed results with confidence scores
- Probability distribution for different skin conditions
- Responsive design for various devices

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone this repository or download the files
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Model Setup

Before running the application, you need to place your trained H5 model file in the `model` directory:

1. Make sure your model file is named `skin_cancer_model.h5`
2. Place it in the `model` directory

If your model has a different name or structure, you may need to modify the `load_model()` function in `app.py`.

## Running the Application

To run the application, execute the following command in the project directory:

```bash
streamlit run app.py
```

The application will start and open in your default web browser. If it doesn't open automatically, you can access it at `http://localhost:8501`.

## Using the Application

1. Upload a clear image of the skin lesion using the file uploader
2. Click the "Analyze Image" button
3. View the results, including:
   - Detected skin condition
   - Confidence score
   - Probability distribution for all possible conditions

## Deployment

This application can be deployed on Streamlit Sharing for free. To deploy:

1. Push your code to a GitHub repository
2. Go to [Streamlit Sharing](https://streamlit.io/sharing)
3. Connect your GitHub account and select the repository
4. Deploy the application

## Disclaimer

This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.

## License

This project is licensed under the MIT License - see the LICENSE file for details."# skin-cancer" 
