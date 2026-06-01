
# CSC173 Deep Computer Vision Project Proposal
**Student:** Chris Adrian D. Gumisad, 2020-3275 
**Date:** 12/11/2025

## 1. Project Title 
A Deep Learning-Based Skin Imperfection Detection System Using MobileNetV2 and Grad-CAM

## 2. Problem Statement
Skin imperfections such as acne marks, dark spots, and discoloration can affect skin appearance and are often difficult to assess consistently through manual observation. Factors such as lighting conditions, image quality, skin tone variations, and subjective human judgment make visual inspection unreliable.

While many beauty and skincare applications provide image enhancement and filtering features, relatively few focus on detecting and analyzing skin imperfections using artificial intelligence. An automated detection system can provide a more objective method for identifying skin conditions and highlighting areas of concern.

This project aims to develop a lightweight computer vision system capable of detecting skin imperfections from facial skin images using deep learning. The system will classify images as either clear skin or skin with imperfections and provide visual explanations using Grad-CAM heatmaps to indicate regions that influenced the model's prediction.

## 3. Objectives
### General Objective
- To develop a lightweight deep learning-based system for detecting skin imperfections using MobileNetV2 and Grad-CAM visualization.

### Specific Objectives
- Build a curated dataset consisting of approximately 2,000 skin images divided into clear skin and skin imperfection classes.
- Apply image preprocessing techniques including resizing, CLAHE enhancement, and normalization.
- Train a MobileNetV2-based binary classification model using transfer learning.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix analysis.
- Measure model inference speed using ONNX Runtime.
- Generate Grad-CAM visualizations to highlight regions contributing to the model’s predictions.
- Develop a Streamlit-based application for local image testing and result visualization.

## 4. Dataset Plan
- **Source:** Handpicked and curated datasets from online sources  
  - Kaggle skin and acne datasets  
  - ISIC skin image datasets
  - DermNet dermatology image collections
  - Manually filtered internet-sourced images  

- **Total Dataset Size:** ~2000 images  

- **Classes:**
  - Clear Skin
  - Skin Imperfection (acne, dark spots, discoloration)

- **Dataset Structure:**
```text
dataset/
    train/
        clear/
        imperfection/

    val/
        clear/
        imperfection/

    test/
        clear/
        imperfection/

- **Acquisition:** 
    - Download datasets using Kaggle
    - Manually review and remove irrelevant or low-quality samples.
    - Resize and preprocess images.
    - Split dataset into training, validation, and test sets
    - Store datasets locally and process them through Google Colab and VSCode environments.

## 5. Technical Approach
### System Pipeline

The system follows a simple deep learning workflow:
    
    Input Image (Skin/Face Image)
    Preprocessing:
        Resize image to 224×224
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Normalize pixel values
    Feature Extraction:
        Use pretrained MobileNetV2 as backbone
    Classification:
        Binary output: Clear Skin or Skin Imperfection
    Explainability:
        Generate Grad-CAM heatmap to highlight important regions
    Output:
        Predicted Class
        Confidence score
        Heatmap overlay visualization
    Model Details:
        Model: MobileNetV2 (pretrained on ImageNet)
        Framework: PyTorch
        Input Size: 224×224 RGB images
        Output: Binary classification
    Tools & Environment:
        Google Colab (training with GPU support)
        VSCode (local testing and development)
        OpenCV (image preprocessing)
        PyTorch (deep learning framework)
        ONNX (optional model optimization for inference)

## 6. Expected Challenges & Mitigations
- Challenge 1 — Limited hardware performance
    Solution: Perform model training using Google Colab GPU resources and use ONNX Runtime for lightweight local inference.

- Challenge 2 — Dataset Quality and Consistency
    Solution: Manually inspect, clean, and verify image labels before training to reduce noise and improve data quality.

- Challenge 3 — Small Dataset Size
    Solution: Maintain balanced class distribution and apply image augmentation techniques to improve model generalization.

- Challenge 4: Model Overfitting
    Solution: Use transfer learning with MobileNetV2, dropout layers, and validation monitoring during training.

- Challenge 5 — Interpretation of Predictions
    Solution: Use Grad-CAM visualization to highlight image regions that contribute to the model’s predictions.