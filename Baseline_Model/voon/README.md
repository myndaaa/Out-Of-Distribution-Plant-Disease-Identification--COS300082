# Model Architecture

- **Model**: A Convolutional Neural Network (CNN) built with Keras.
- **Input Shape**: RGB images with dimensions `299x299x3`.

### Layers:
1. **Input Layer**:
   - Shape: `(299, 299, 3)`.

2. **Convolutional Layers**:
   - `Conv2D` with 32 filters, kernel size `(3, 3)`, activation `ReLU`.
   - `Conv2D` with 64 filters, kernel size `(3, 3)`, activation `ReLU`.
   - `Conv2D` with 128 filters, kernel size `(3, 3)`, activation `ReLU`.

3. **Pooling Layers**:
   - `MaxPooling2D` after each convolutional layer, pool size `(2, 2)`.

4. **Flatten Layer**:
   - Flattens feature maps into a single vector.

5. **Dense Layers**:
   - `Dense` with 256 units, activation `ReLU`.
   - `Dense` with `num_classes` units (14 classes for crop classification), activation `softmax`.

6. **Dropout Layer**:
   - Dropout rate of 0.5 after the first dense layer to prevent overfitting.

### Model Details:
- **Optimizer**: Adam, with a learning rate of 0.001.
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy.
 

# Results
## Metrics
-   **Accuracy**: Measures the correct predictions across all classes.
-   **Validation Accuracy**: Accuracy on the seen test dataset during training.
-   **Top-1 Accuracy**: Accuracy when the correct class is the top prediction.
-   **Top-5 Accuracy**: Accuracy when the correct class is within the top 5 predictions.
-   **Harmonic Mean Accuracy**: A balanced metric combining accuracy on seen and unseen datasets.
- ### Evaluation

-   **Training Set**:
    -   Accuracy: 94.55%.
    -   Validation Accuracy: 96.98%.
-   **PV Seen Test Set**:
    -   Accuracy: 96.98%.
-   **PV Unseen Test Set**:
    -   Top-1 Accuracy: 12.56%.
    -   Top-5 Accuracy: 78.60%.
-   **PD Unseen Test Set**:
    -   Accuracy: 8.57%.
-   **Harmonic Mean** (PV Seen vs PV Unseen): 22.24%.

## Key Features

-   **Data Augmentation**: Applied to training data to improve model generalization:
    -   Rotation, flipping, zoom, shear, brightness adjustments, etc.
-   **Callbacks**:
    -   ReduceLROnPlateau: Dynamically reduces learning rate on a plateau in validation loss.
    -   EarlyStopping: Halts training if validation loss does not improve after a set number of epochs.
-   **Save and Load Model**:
    -   Models are saved in `.h5` format for compatibility and easy loading.
