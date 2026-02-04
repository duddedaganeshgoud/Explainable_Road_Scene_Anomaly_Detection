import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import shap
import matplotlib.pyplot as plt
import pickle
import time
from PIL import Image
import io
import traceback

# Set page config for a wider layout
st.set_page_config(
    page_title="Road Feature Classifier",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
IMG_HEIGHT = 160
IMG_WIDTH = 160
CHANNELS = 3
CLASS_NAMES = ['HMV', 'LMV', 'Pedestrian', 'RoadDamages', 'SpeedBump', 'UnsurfacedRoad']
CLASS_DESCRIPTIONS = {
    'HMV': 'Heavy Motor Vehicle',
    'LMV': 'Light Motor Vehicle',
    'Pedestrian': 'Person walking on or near the road',
    'RoadDamages': 'Road with cracks, potholes or other damage',
    'SpeedBump': 'Speed bumps or traffic calming measures',
    'UnsurfacedRoad': 'Dirt or gravel road without paving'
}

# Model accuracy information
MODEL_ACCURACY = {
    'YOLO': 0.80,
    'EfficientNet': 0.23,
    'ResNet50': 0.75,
    'VGG16': 0.94,
    'DenseNet121': 0.93,
    'Custom CNN': 0.77
}

# Functions to load models and explainers
@st.cache_resource
def load_vgg16_model():
    """Load the saved VGG16 model"""
    model_path = 'final_vgg16_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check the path.")
        return None
    
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return None

@st.cache_resource
def load_background_data():
    """Load the saved background data for SHAP explainer"""
    bg_path = 'shap_background.pkl'
    if not os.path.exists(bg_path):
        st.warning(f"Background data file not found at {bg_path}")
        return None
    
    try:
        with open(bg_path, 'rb') as f:
            background = pickle.load(f)
        
        st.success(f"Loaded background data with shape {background.shape}")
        return background
    except Exception as e:
        st.warning(f"Error loading background data: {str(e)}")
        return None

@st.cache_resource
def create_explainer_with_background(model, background):
    """Create a SHAP explainer with provided background data"""
    try:
        if background is None:
            st.warning("No background data provided for creating explainer")
            return None
            
        # Define model wrapper
        def model_predict(images):
            return model.predict(images)
        
        # Create a masker using the first background sample
        masker = shap.maskers.Image("inpaint_telea", background[0].shape)
        
        # Create explainer
        explainer = shap.Explainer(model_predict, masker, output_names=CLASS_NAMES)
        st.success("Created SHAP explainer from background data")
        
        return explainer
    except Exception as e:
        st.warning(f"Error creating SHAP explainer from background: {str(e)}")
        return None

@st.cache_resource
def create_simple_explainer(model):
    """Create a simple SHAP explainer without background data"""
    try:
        def model_predict(images):
            return model.predict(images)
        
        # Create inpainting masker that doesn't need background data
        masker = shap.maskers.Image("inpaint_telea", (IMG_HEIGHT, IMG_WIDTH, CHANNELS))
        
        # Create explainer
        explainer = shap.Explainer(model_predict, masker, output_names=CLASS_NAMES)
        st.success("Created simple SHAP explainer (no background data)")
        
        return explainer
    except Exception as e:
        st.error(f"Error creating simple SHAP explainer: {str(e)}")
        return None

def preprocess_image(img):
    """Preprocess a PIL image for model prediction"""
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

def predict_image(model, img_array):
    """Run prediction on the preprocessed image"""
    prediction = model.predict(img_array)
    pred_class_idx = np.argmax(prediction[0])
    pred_class = CLASS_NAMES[pred_class_idx]
    confidence = prediction[0][pred_class_idx]
    
    # If confidence is very low (below 0.4), consider it as plain road
    if confidence < 0.4:
        pred_class = "Plain Road (low confidence)"
        all_probs = {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))}
        all_probs["Plain Road"] = 1.0 - confidence
        return pred_class, confidence, all_probs
    
    # Get all class probabilities
    all_probs = {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))}
    
    return pred_class, confidence, all_probs

def generate_shap_explanation(explainer, img_array, pred_class_idx):
    """Generate SHAP explanation for the image"""
    try:
        # Calculate SHAP values
        shap_values = explainer(img_array)
        
        # Handle different SHAP output formats
        if hasattr(shap_values, 'values'):
            if shap_values.values.ndim == 5:  # (samples, h, w, c, classes)
                shap_image_values = shap_values.values[0, :, :, :, pred_class_idx]
            else:  # (samples, h, w, c)
                shap_image_values = shap_values.values[0]
        else:
            # For older SHAP versions with numpy arrays
            if isinstance(shap_values, list):
                shap_image_values = shap_values[pred_class_idx][0]
            else:
                shap_image_values = shap_values[0]
        
        # Create a matplotlib figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(img_array[0])
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # SHAP explanation
        abs_shap_values = np.abs(shap_image_values).sum(axis=-1)
        im = ax2.imshow(abs_shap_values, cmap='hot')
        ax2.set_title("Features Influencing Prediction")
        ax2.axis('off')
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Feature Importance')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
        traceback.print_exc()
        return None

# App title and description
st.title("🛣️ Road Feature Classification")
st.markdown("""
This app uses a VGG16 deep learning model to classify road features from images. 
Upload an image to get a prediction and see which parts of the image influenced the classification.
""")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("This tool helps identify road features using computer vision.")
    
    st.header("Model Performance")
    st.markdown("Accuracy comparison of different models:")
    
    # Create a bar chart of model accuracies
    fig, ax = plt.subplots(figsize=(8, 4))
    models = list(MODEL_ACCURACY.keys())
    accuracies = list(MODEL_ACCURACY.values())
    
    # Highlight VGG16
    colors = ['#1f77b4' if model != 'VGG16' else '#ff7f0e' for model in models]
    
    bars = ax.bar(models, accuracies, color=colors)
    ax.set_ylabel('MAP / Accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_title('Model Accuracy Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    st.markdown("**VGG16 model** (accuracy: 94%) is used for predictions in this app.")
    
    st.header("Class Information")
    for class_name in CLASS_NAMES:
        st.markdown(f"**{class_name}**: {CLASS_DESCRIPTIONS[class_name]}")

# Load the model
model = load_vgg16_model()

if model is None:
    st.error("Failed to load the model. Please check the model path.")
    st.stop()

# Create SHAP explainer using a more reliable approach
# First, try to load background data
background = load_background_data()

# Create explainer with background data if available
if background is not None:
    st.info("Creating SHAP explainer with background data...")
    explainer = create_explainer_with_background(model, background)
else:
    # If background data not available, create simple explainer
    st.info("Background data not available. Creating simple SHAP explainer...")
    explainer = create_simple_explainer(model)

if explainer is None:
    st.warning("Could not create SHAP explainer. XAI features will not be available.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image of a road feature", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and make prediction
    with st.spinner("Processing image..."):
        img_array = preprocess_image(image_pil)
        pred_class, confidence, all_probs = predict_image(model, img_array)
        
        # Get the proper index for the SHAP explanation
        if "Plain Road" in pred_class:
            # If it's a plain road (low confidence), use the highest probability class for SHAP
            pred_class_idx = np.argmax([all_probs[cls] for cls in CLASS_NAMES])
        else:
            pred_class_idx = CLASS_NAMES.index(pred_class)
    
    # Display prediction results
    with col2:
        st.subheader("Classification Results")
        
        # Display the prediction
        if "Plain Road" in pred_class:
            st.markdown(f"### Prediction: **{pred_class}**")
            st.markdown("The model has low confidence in its prediction, suggesting this might be a plain road without special features.")
        else:
            st.markdown(f"### Prediction: **{pred_class}**")
            if pred_class in CLASS_DESCRIPTIONS:
                st.markdown(f"*{CLASS_DESCRIPTIONS[pred_class]}*")
        
        # Display confidence
        st.markdown(f"**Confidence:** {confidence:.2%}")
        
        # Create a bar chart for all class probabilities
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = list(all_probs.keys())
        probs = list(all_probs.values())
        
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]  # Descending order
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_probs = [probs[i] for i in sorted_indices]
        
        # Use different colors for the predicted class
        colors = ['#1f77b4' if cls != pred_class else '#ff7f0e' for cls in sorted_classes]
        
        ax.barh(sorted_classes, sorted_probs, color=colors)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1.0)
        ax.set_title('Class Probabilities')
        
        # Add text labels for probabilities
        for i, v in enumerate(sorted_probs):
            ax.text(v + 0.01, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Generate and display SHAP explanation
    if explainer is not None:
        st.subheader("Explainable AI (XAI) Analysis")
        st.markdown("""
        The heatmap below shows which parts of the image influenced the model's prediction.
        Brighter areas had more influence on the classification decision.
        """)
        
        with st.spinner("Generating SHAP explanation..."):
            shap_fig = generate_shap_explanation(explainer, img_array, pred_class_idx)
            if shap_fig:
                st.pyplot(shap_fig)
            else:
                st.warning("Could not generate SHAP explanation for this image.")
    else:
        st.error("No SHAP explainer available. XAI analysis cannot be performed.")
    
    # Technical details (collapsible)
    with st.expander("Technical Details"):
        st.markdown("""
        **Model Architecture:** VGG16 (Modified)  
        **Input Size:** 160x160x3  
        **Classes:** HMV, LMV, Pedestrian, RoadDamages, SpeedBump, UnsurfacedRoad  
        **XAI Method:** SHAP (SHapley Additive exPlanations)
        """)
        
        # Show preprocessing steps
        st.code("""
# Preprocessing steps:
1. Resize image to 160x160 pixels
2. Normalize pixel values (divide by 255)
3. Pass through VGG16 model for prediction
4. Generate SHAP values to explain the prediction
        """)
else:
    # Display sample images when no file is uploaded
    st.info("👆 Upload an image to get started!")

# Footer
st.divider()
st.markdown("""
**Note:** This app uses a pre-trained VGG16 model for road feature classification. 
For accurate results, upload clear images of roads or road features.
""")