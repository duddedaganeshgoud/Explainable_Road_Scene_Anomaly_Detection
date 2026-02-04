import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from tqdm import tqdm

# Constants
IMG_HEIGHT = 160
IMG_WIDTH = 160
CHANNELS = 3
BATCH_SIZE = 8
CLASS_NAMES = ['HMV', 'LMV', 'Pedestrian', 'RoadDamages', 'SpeedBump', 'UnsurfacedRoad']
OUTPUT_DIR = 'shap_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to your dataset
TRAIN_PATH = 'organized_dataset/test'
BACKGROUND_DATA_PATH = os.path.join(OUTPUT_DIR, 'shap_background.pkl')

def load_images_from_folder(folder, num_samples=None):
    """Load and preprocess images from a folder"""
    datagen = ImageDataGenerator(rescale=1./255)
    parent_dir = os.path.dirname(folder)
    class_name = os.path.basename(folder)

    print(f"Loading images from {folder}")
    generator = datagen.flow_from_directory(
        parent_dir,
        classes=[class_name],
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=True
    )

    images = []
    batches_needed = int(np.ceil(num_samples/BATCH_SIZE)) if num_samples else 1
    for i, batch in enumerate(generator):
        images.extend(batch)
        if num_samples and len(images) >= num_samples:
            break
        if i >= batches_needed:
            break

    images = np.array(images[:num_samples] if num_samples else images)
    print(f"Loaded {len(images)} images with shape {images.shape}")
    return images

def get_background_samples(num_samples=50):
    """Get balanced background samples"""
    background = []
    samples_per_class = max(1, num_samples // len(CLASS_NAMES))

    for class_name in tqdm(CLASS_NAMES, desc="Collecting background samples"):
        class_path = os.path.join(TRAIN_PATH, class_name)
        if os.path.exists(class_path):
            class_images = load_images_from_folder(class_path, samples_per_class)
            if len(class_images) > 0:
                background.extend(class_images)
        else:
            print(f"Warning: Class folder not found: {class_path}")

    background = np.array(background)
    print(f"Collected {len(background)} background samples")
    return background

if __name__ == "__main__":
    print("Starting background data generation")
    
    # Check if background data already exists
    if os.path.exists(BACKGROUND_DATA_PATH):
        print(f"Background data already exists at {BACKGROUND_DATA_PATH}. Delete it if you want to regenerate.")
    else:
        # Generate background data
        print("Creating new background samples...")
        background = get_background_samples(50)
        print("Background shape:", background.shape)
        
        # Save background data
        print(f"Saving background data to {BACKGROUND_DATA_PATH}")
        with open(BACKGROUND_DATA_PATH, 'wb') as f:
            pickle.dump(background, f)
        
        print(f"Background data saved to {BACKGROUND_DATA_PATH}")
    
    print("Done!")