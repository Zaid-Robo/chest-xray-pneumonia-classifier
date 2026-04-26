# Technical Report: Chest X-Ray Pneumonia Classification

## 1. Executive Summary
This project implements a complete, production-aware machine learning pipeline for binary classification of chest X-ray images to detect pneumonia. Following a robust ML engineering workflow, we explored data characteristics, established a baseline, and developed a high-performing fine-tuned model. The final solution is containerized and ready for public deployment via a Gradio-based web interface.

## 2. Problem Framing
The task is defined as a binary image classification problem:
- **Input**: Anterior-posterior (AP) chest X-ray image.
- **Output**: Class label (NORMAL or PNEUMONIA) with associated confidence scores.

**Objective**: Maximize the Recall (Sensitivity) for the PNEUMONIA class to minimize false negatives in a medical context, while maintaining high overall precision.

## 3. Data Understanding & Preparation
### Dataset Choice
We utilized the `hf-vision/chest-xray-pneumonia` dataset (hosted on HuggingFace). 
- **Size**: ~5,800 images.
- **Distribution**: Significant class imbalance (~3:1 ratio of Pneumonia to Normal).
- **Format**: High-resolution X-ray images with varying aspect ratios.

### Preprocessing & Augmentation
To improve model generalization and handle variability:
- **Resizing**: All images normalized to 224x224 pixels.
- **Augmentation**: Random horizontal flips, rotations (±10°), and color jittering were applied to the training set to simulate variations in imaging conditions.
- **Imbalance Handling**: We implemented a dual strategy:
  1. **WeightedRandomSampler**: Ensures each batch in training is balanced.
  2. **Class-Weighted Cross-Entropy Loss**: Penalizes errors on the minority (NORMAL) class more heavily.

## 4. Model Selection & Training Strategy
We compared two distinct architectures to demonstrate engineering judgment:

### Model A: SimpleCNN (Baseline)
- **Architecture**: A lightweight 3-layer convolutional network built from scratch.
- **Purpose**: To establish a performance floor and verify the integrity of the data pipeline.
- **Result**: Demonstrated basic learning capacity but lacked the depth to capture complex pathological textures.

### Model B: MobileNetV2 (Production Model)
- **Architecture**: A depthwise separable convolutional network pretrained on ImageNet.
- **Strategy**: 
  - **Phase 1 (Warm-up)**: Frozen backbone, training only the custom classification head.
  - **Phase 2 (Fine-tuning)**: Gradual unfreezing of the backbone with a significantly reduced learning rate.
- **Justification**: MobileNetV2 was chosen for its extreme efficiency on CPU/constrained hardware (only ~3.4M parameters) while maintaining high accuracy, directly reflecting the "compute-aware" constraint of the assessment.

## 5. Evaluation & Error Analysis
### Quantitative Results
- **Metrics**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
- **Performance**: The fine-tuned MobileNetV2 significantly outperformed the baseline, achieving high sensitivity on the PNEUMONIA class.

### Qualitative Analysis
Using the generated `evaluate.py` script, we identified common failure modes:
- **False Positives**: Normal lungs with low-contrast or significant overlapping shadows (e.g., rib cage density) occasionally triggered pneumonia predictions.
- **False Negatives**: Early-stage pneumonia with very subtle "ground-glass" opacities was the most common error.

## 6. Engineering Quality & Deployment
### Code Structure
The project follows a modular structure beyond a single notebook:
- `src/`: Core logic (data loaders, model definitions, training/eval loops).
- `configs/`: Centralized YAML configuration.
- `app.py`: Production inference interface using Gradio.

### Deployment Readiness
The model is exposed via a **Gradio web interface** and is fully containerized using a **Dockerfile**. 
- **Live Link**: [Insert your deployment URL here]

## 7. Scaling & Future Improvements
Given unlimited compute, we would explore:
1. **Architectural Scaling**: EfficientNet-B3 or Vision Transformers (ViT).
2. **Interpretability**: Integration of Grad-CAM to highlight suspicious regions for clinicians.
3. **Robustness**: 5-fold cross-validation and ensembling of multiple checkpoints.
