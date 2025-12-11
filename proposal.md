
# CSC173 Deep Computer Vision Project Proposal
**Student:** Chris Adrian D. Gumisad, 2020-3275 
**Date:** 12/11/2025

## 1. Project Title 
A Computer Vision-Based System for Detecting Skin Imperfections Using Color Segmentation and Deep Learning

## 2. Problem Statement
Skin imperfections such as dark spots, acne marks, and discoloration are one of the main factors that keep people from their self confidence in public interaction or even private setting. It is often difficult to detect consistently due to variations like lighting, skin tone, and human perception. We can also do manual inspection but it is also subjective and prone to errors, leading to missed imperfections that are relevant for skincare monitoring or makeup application. Existing mobile beauty tools typically focus on enhancement rather than accurate detection. This project aims to create a reliable computer vision system capable of objectively identifying skin imperfections using color segmentation and deep learning, providing users an accessible tool for monitoring skin clarity and progress over time.

## 3. Objectives
### General Objective
- To develop a lightweight and cloud-assisted computer vision system capable of detecting skin imperfections using color segmentation and deep learning.

### Specific Objectives
- Train a CNN model using cloud GPUs (Colab/Kaggle) to identify dark spots, acne marks, and discoloration.
- Implement preprocessing (normalization, CLAHE, color space conversion) that runs efficiently even on low-end devices.
- Use online/cloud platforms to handle model training, dataset processing, and evaluation.
- Generate accurate visual outputs (heatmaps/overlays) without requiring heavy local computation.
- Develop a lightweight demo app/interface that runs on a low end laptop.

## 4. Dataset Plan
- Source: Use online datasets (no local storage required):
    - Kaggle: Acne Detection Dataset
    - ISIC Skin Lesion Archive
    - DermNet pigment/dark spot images
Expected image count: 5,000–15,000 images
Can only preview samples locally; processing and training will be done in the cloud.

- Classes: 
    - Dark Spots
    - Acne Marks
    - Hyperpigmentation
    - Clear Skin (negative class)

- Acquisition: 
    - Import datasets directly from sources into Google Colab (kaggle.json API).
    - Preprocess images in Colab notebooks.
    - No need to download the dataset to your laptop.

## 5. Technical Approach
- Architecture sketch
    - Cloud-based training (Google Colab GPU)
    - Color segmentation using OpenCV
    - CNN inference with optimized model (MobileNetV2)
    - Visualization (overlays/heatmaps)

- Model: MobileNetV2

- Framework: 
    - Training: Google Colab → PyTorch
    - Local Testing: ONNX Runtime
    - Image Processing: OpenCV

- Hardware: Google Colab GPU (T4/A100) and Kaggle Notebook GPU

## 6. Expected Challenges & Mitigations
- Challenge 1 — Weak laptop hardware
    Solution: 
    Perform all training and heavy processing in:
        - Google Colab
        - Kaggle Notebooks
        - HuggingFace Spaces

- Challenge 2 — Dataset too large for your storage
    Solution: Load dataset directly in Colab via Kaggle API

- Challenge 3 — Heavy CNN models slow on CPU
    Solution: 
    - Use MobileNetV2 or EfficientNet-B0
    - Export to ONNX for fast inference
    - Use small image sizes (224x224)

- Challenge 4 — Cloud GPU time limits
    Solution:
    - Use Kaggle GPU as backup
    - Save checkpoints to Google Drive
    - Use smaller batch sizes and mixed precision

- Challenge 5 — Internet dependency
    Solution:
    - Cache datasets in Colab
    - Keep notebooks synced in Drive
    - Export model only once for local use