# Explainable Road Scene Anomaly Detection

This repository presents an **Explainable Deep Learning framework for Road Scene Anomaly Detection**, combining object detection and post-hoc explainability techniques to improve transparency and trust in intelligent traffic monitoring systems.

The system detects anomalous events in road scenes (such as accidents, unusual objects, or abnormal traffic behavior) and provides **visual explanations** to justify the model’s predictions.

---

## 📌 Motivation

Traditional deep learning models for traffic anomaly detection often behave as **black boxes**, making it difficult to understand *why* a particular scene is classified as anomalous.  
This lack of interpretability limits their adoption in **safety-critical applications** such as intelligent transportation systems.

This project addresses the problem by integrating **explainable AI (XAI)** techniques with deep learning-based anomaly detection.

---

## 🧠 Proposed Approach

The proposed framework consists of three major stages:

1. **Feature Extraction & Detection**
   - A YOLO-based object detection model is used to extract spatial and semantic information from road scenes.
   - Deep CNN architectures (VGG16 / ResNet50) are employed to learn high-level representations.

2. **Anomaly Detection**
   - The extracted features are used to distinguish between *normal* and *anomalous* road scenes.
   - The model is trained on normal traffic patterns and flags deviations as anomalies.

3. **Explainability**
   - SHAP (SHapley Additive exPlanations) is applied to generate interpretable explanations.
   - Visual explanations highlight which regions and features contributed most to the anomaly decision.

---

## 🗂️ Project Structure

