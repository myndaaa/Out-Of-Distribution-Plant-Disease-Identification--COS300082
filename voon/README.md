Model Architecture
The model architecture is built using a Convolutional Neural Network (CNN) and consists of the following layers:

Input Layer: Accepts images of shape (299, 299, 3).
Three convolutional blocks:
Each block includes convolutional layers (Conv2D), followed by max-pooling layers (MaxPooling2D).
Activation: ReLU.
Fully connected dense layer with 256 units.
Dropout layer with a rate of 0.5 to prevent overfitting.
Output Layer: A dense layer with 14 units (softmax activation) for multi-class classification.
Optimizer: Adam, with a learning rate of 0.001.
Loss Function: Categorical Crossentropy.
Metrics: Accuracy.

Results
Metrics
Accuracy: Measures the correct predictions across all classes.
Validation Accuracy: Accuracy on the seen test dataset during training.
Top-1 Accuracy: Accuracy when the correct class is the top prediction.
Top-5 Accuracy: Accuracy when the correct class is within the top 5 predictions.
Harmonic Mean Accuracy: A balanced metric combining accuracy on seen and unseen datasets.
Evaluation
Training Set:
Accuracy: 94.55%.
Validation Accuracy: 96.98%.
PV Seen Test Set:
Accuracy: 96.98%.
PV Unseen Test Set:
Top-1 Accuracy: 12.56%.
Top-5 Accuracy: 78.60%.
PD Unseen Test Set:
Accuracy: 8.57%.
Harmonic Mean (PV Seen vs PV Unseen): 22.24%.

Key Features
Data Augmentation: Applied to training data to improve model generalization:
Rotation, flipping, zoom, shear, brightness adjustments, etc.
Callbacks:
ReduceLROnPlateau: Dynamically reduces learning rate on a plateau in validation loss.
EarlyStopping: Halts training if validation loss does not improve after a set number of epochs.
Save and Load Model:
Models are saved in .h5 format for compatibility and easy loading.
