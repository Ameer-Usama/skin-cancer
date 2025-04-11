import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for the skin cancer detection model.
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) to resize the image to
        
    Returns:
        Preprocessed image array ready for model prediction
    """
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


def get_gradcam(model, img_array, layer_name='conv2d_1'):
    """
    Generate Grad-CAM visualization for the given image.
    
    Args:
        model: Loaded TensorFlow model
        img_array: Preprocessed image array
        layer_name: Name of the layer to use for Grad-CAM
        
    Returns:
        Heatmap overlay on the original image
    """
    try:
        # Get the score for the predicted class
        with tf.GradientTape() as tape:
            # Get the target layer
            grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(layer_name).output, model.output]
            )
            
            # Get the activations of the last conv layer and predictions
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Gradient of the predicted class with respect to the output feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by corresponding gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match the original image size
        img = img_array[0] * 255
        img = img.astype(np.uint8)
        
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap to heatmap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on original image
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None


def get_image_download_link(img, filename="skin_analysis.jpg", text="Download Result"):
    """
    Generate a download link for an image.
    
    Args:
        img: PIL Image or numpy array
        filename: Name of the file to download
        text: Text to display for the download link
        
    Returns:
        HTML string with the download link
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    
    # Create a byte buffer
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    
    # Encode the bytes to base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create the HTML download link
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    
    return href


def plot_prediction_bars(probabilities, class_names):
    """
    Create a horizontal bar chart of prediction probabilities.
    
    Args:
        probabilities: Dictionary of class probabilities
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    # Sort probabilities
    sorted_pairs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [x[0] for x in sorted_pairs]
    probs = [x[1] for x in sorted_pairs]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bars
    bars = ax.barh(classes, probs, color=['red' if c == 'Melanoma' else '#0083B8' for c in classes])
    
    # Add percentage labels to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_position = width + 0.01
        ax.text(label_position, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                va='center', fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')
    ax.set_xlim(0, 1.15)  # Add some space for the percentage labels
    plt.tight_layout()
    
    return fig