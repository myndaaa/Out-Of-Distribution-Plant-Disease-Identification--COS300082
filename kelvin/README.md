# Plant Disease Classification Project

## Overview

This project implements a CNN-based classification model for plant diseases. The goal is to classify images into one of 21 disease categories across 14 crop categories, using training and testing datasets from both PlantVillage (PV) and PlantDoc (PD).

---

## Architecture

- **Model**: A Convolutional Neural Network (CNN) built with Keras.
- **Input Shape**: RGB images with dimensions 299x299x3.
- **Layers**:
  1. **Input Layer**: Shape `(299, 299, 3)`.
  2. **Convolutional Layers**:
     - `Conv2D` with 32 filters, kernel size `(3, 3)`, activation `ReLU`.
     - `Conv2D` with 64 filters, kernel size `(3, 3)`, activation `ReLU`.
     - `Conv2D` with 128 filters, kernel size `(3, 3)`, activation `ReLU`.
  3. **Pooling Layers**: 
     - `MaxPooling2D` after each convolutional layer, pool size `(2, 2)`.
  4. **Flatten Layer**: Flattens feature maps into a single vector.
  5. **Dense Layers**:
     - `Dense` with 256 units, activation `ReLU`.
     - `Dense` with `num_classes` units (21 classes), activation `softmax`.
  6. **Dropout Layer**: Dropout rate of 0.5 after the first dense layer to prevent overfitting.
- **Loss Function**: Categorical Cross-Entropy.
- **Optimizer**: RMSprop.
- **Learning Rate Scheduling**: Reduces learning rate when validation accuracy plateaus.
- **Epochs**: 15.

---

## Results

### Training and Validation Metrics:
- **Final Training Accuracy**: 90.54%
- **Final Validation Accuracy**: 94.36%

### Key Metrics:
- **Top-1 Accuracy**: 10.23%
- **Top-5 Accuracy**: 60.00%
- **Harmonic Mean Accuracy (Seen & Unseen)**: 18.46%
- **Unseen PV Test Accuracy**: 10.23%
- **Seen PV Test Accuracy**: 94.37%
- **PlantDoc Test Accuracy**: 12.86%


---

## Datasets

### PlantVillage (PV)
- **Train Set**: Used for model training.
- **Seen Test Set**: Evaluates performance on disease classes present during training.
- **Unseen Test Set**: Evaluates performance on unseen disease categories.

### PlantDoc (PD)
- **Unseen Test Set**: Real-world images for evaluation under field conditions.



