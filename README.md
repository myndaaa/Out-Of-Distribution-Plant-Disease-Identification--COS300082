[![Typing SVG](https://readme-typing-svg.demolab.com?font=Sour+Gummy&weight=900&size=30&pause=1000&color=E6EAFF&background=4C186A&center=true&width=800&height=90&lines=Out+of+Distribution+Plant+Disease+Identification)](https://git.io/typing-svg)

## Dataset Overview and Project Focus

In this repository, you can find the datasets used in our project on **Out-of-Distribution (OOD) and Skewed Data**, 
and the exploration of **Zero-Shot Learning** as a possible technique to address these challenges.

The datasets can be accessed via the following link:  
[Dataset Link - Google Drive](https://drive.google.com/drive/folders/1atwfzhMEn2P4jwJqeQUJjpvfdFT1tS_v)

Our project aims to delve into how OOD and skewed data can impact machine learning models and 
investigate strategies, such as **Zero-Shot Learning**, to mitigate these issues.

---

### Dataset Breakdown

The datasets consist of the following:

| **Dataset**      | **Train Images** | **Test Images** | **Total Images** |
|------------------|------------------|-----------------|------------------|
| **PlantVillage** | 38,994           | 10,495          | 49,489           |
| **PlantDoc**     | -                | 71              | 71               |

---

### Dataset File Contents

The dataset contains several CSV files with metadata related to the images. These files are structured as follows:

| **File/Folder**                | **Content**                                                                                              |
|---------------------------------|----------------------------------------------------------------------------------------------------------|
| `PV train.csv`                  | List of laboratory training images with ground truth from the PlantVillage dataset.                       |
| `PV test seen.csv`              | List of seen testing images with ground truth from the PlantVillage dataset.                             |
| `PV test unseen.csv`            | List of unseen testing images with ground truth from the PlantVillage dataset.                           |
| `PD test unseen.csv`            | List of unseen testing images with ground truth from the PlantDoc dataset.                               |

Each CSV file contains the following columns:

1. **Image File Name**: The name of the image file.
2. **Crop Category (Ground Truth)**: The category of the crop (there are 14 crop categories).
3. **Disease Category (Ground Truth)**: The category of the disease affecting the crop (there are 21 disease categories).

The crop and disease categories are combined to form a single composition. The index for both crop and disease categories starts from 0.

---

# Baseline Model 🌿


### 1. **Preprocessing:**
   - **Image Size:** All training images are resized to 256 x 256 pixels.
   - **Normalization:** Pixel values are scaled to the range [0, 1] using ImageNet's mean and standard deviation values for normalization:
     - **Mean:** [0.485, 0.456, 0.406]
     - **Std:** [0.229, 0.224, 0.225]
   
### 2. **Data Augmentation:**
   To simulate real-world variations, the following augmentations are applied:
   - Random rotation, flips, zoom, shifts (width & height), brightness/contrast adjustments.
   - Random cropping and padding.

### 3. **Class Imbalance Handling:**
   - **Oversampling** of smaller classes using techniques like SMOTE.
   - **Class Weights:** Higher weights are assigned to underrepresented classes to balance model learning.

### 4. **Transfer Learning with ResNet:**
   - **Frozen Layers:** Initially, the convolutional layers of the pre-trained ResNet model are frozen.
   - **Custom Dense Layers:** Added on top of the base model for classification.
   - **Fine-Tuning:** Later, the top layers of ResNet are unfrozen and fine-tuned with a low learning rate.

### 5. **Training Phases:**
   - **Phase 1:** Train custom layers only, freezing the pre-trained layers.
   - **Phase 2:** Fine-tune the entire model with a low learning rate and early stopping to prevent overfitting.

### 6. **Regularization & Optimization:**
   - **Dropout Layers** to prevent overfitting.
   - **L2 Regularization** on weights.
   - **Optimizer:** Adam or RMSprop with a learning rate schedule.
   - **Loss Function:** Categorical Cross-Entropy for multi-class classification.

### 7. **Model Evaluation:**
   - After training, evaluate model performance on a separate test set.
   - **Metrics:** Accuracy, precision, recall, and F1-score.

### 8. **Hyperparameter Tuning:**
   - Use grid search/random search for optimizing hyperparameters like learning rate, batch size, and epochs.
   - **Ensemble Methods:** Experiment with boosting techniques to further increase model accuracy.

---

By: Mysha, Voon, Kelvin
