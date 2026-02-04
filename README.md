# Explainable Road Scene Anomaly Detection Using VGG16 and XAI

This repository contains the implementation of the research work  
**“Explainable Road Scene Anomaly Detection Using VGG16 and XAI”**,  
which proposes a transparent deep learning framework for automated road scene anomaly classification using **VGG16** and **SHAP-based Explainable AI (XAI)**.

The system is designed to improve **trust, transparency, and reliability** in intelligent transportation and traffic surveillance systems by providing **pixel-level explanations** for model predictions.

---

## Abstract

Smart transportation systems require automated and reliable road scene understanding.  
While deep learning models achieve high accuracy, they often function as **black boxes**, limiting their adoption in safety-critical environments.

This project presents an **Explainable Road Scene Anomaly Detection system** that integrates:
- **VGG16 deep learning architecture** for feature extraction and classification
- **SHAP (SHapley Additive exPlanations)** for pixel-level interpretability

The system classifies six critical road scene categories:
- Heavy Motor Vehicles (HMV)
- Light Motor Vehicles (LMV)
- Pedestrians
- Road Damages
- Speed Bumps
- Unsurfaced Roads

The model achieves approximately **94% classification accuracy** and provides visual explanations that highlight the discriminative regions influencing each prediction.  
An interactive **Streamlit-based interface** enables real-time image upload, inference, and explainability visualization.

---

## Keywords

Explainable AI (XAI), VGG16, SHAP, Road Scene Classification, Deep Learning, Computer Vision, Streamlit, Transportation Monitoring

---

## Methodology Overview

The proposed system follows a four-stage pipeline:

1. **Data Preprocessing**
   - Input road scene images are resized to a fixed resolution
   - Pixel normalization is applied for numerical stability

2. **Feature Extraction Using VGG16**
   - VGG16 is employed as the backbone CNN due to its strong hierarchical feature learning capability
   - Fully connected layers and a softmax classifier are used for multi-class classification

3. **Model Training and Inference**
   - The model is trained using categorical cross-entropy loss and Adam optimizer
   - Early stopping is applied to prevent overfitting
   - During inference, the model predicts one of six road scene categories

4. **Explainability Using SHAP**
   - SHAP generates pixel-level attribution heatmaps
   - Visual explanations reveal which image regions contributed most to the prediction
   - This improves transparency and auditability of the system

---

## Dataset Information

The system is trained and evaluated using the **RAD (Road Anomaly Detection) dataset**, which contains:

- ~11,800 labeled images
- 371 Indian road scenarios
- Six road scene classes:
  - LMV
  - HMV
  - Pedestrians
  - Road Damages
  - Unsurfaced Roads
  - Speed Bumps

Due to GitHub storage limitations, the dataset is **not included** in this repository.

---

## Project Structure

Explainable_Road_Scene_Anomaly_Detection/
│
├── EXPLAINABLE_TRAFFIC_ANOMALY/
│ ├── app.py # Streamlit application for inference
│ ├── background.py # Background processing utilities
│ ├── 874.ipynb # Model training and experimentation
│ ├── 874-yolo.ipynb # Comparative YOLO experiments
│ ├── requirements.txt # Python dependencies
│ │
│ ├── shap_results/
│ │ └── shap_background.pkl # SHAP background data
│ │
│ ├── yolo/
│ │ └── kaggle/working/ # YOLO utilities (comparison only)
│
├── .gitignore # Excludes models, datasets, outputs
├── README.md # Project documentation
