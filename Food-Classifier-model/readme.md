# Food Detector Model Directory

This directory contains all the code and files related to the machine learning model used for food identification in the Food Detector iOS app.

## Files and Directories

### Model Training and Testing

#### `model.py`

This is the main script that contains the architecture and training logic for the food detection model. It defines the neural network architecture and training loops.

#### `test-model.py`

This Python script is used for testing the trained model on custom images. It loads the model and predicts the food items based on the given images.

### Model Checkpoints

#### `checkpoint_8.pth` and `checkpoint_5.pth`

These files are the saved states (checkpoints) of the trained model. These are the best-performing versions of the model achieved during training.

### Model Conversion

#### `coreml_converter.py`

This script is responsible for converting the trained PyTorch model into a Core ML model format so that it can be used in the iOS app.

### Dataset Preprocessing

#### `ms_calculator.py`

This script calculates the mean and normalization values for the images in the dataset. These values are used during both training and inference to standardize the input images.

### Labels and Classes

#### `labels.txt`

Contains the labels for the food items that the machine learning model can identify. This file is used in the iOS app to map prediction indices to human-readable labels.

#### `classes.txt`

Contains the class names corresponding to the labels. This file is primarily used during the training and testing of the machine learning model.
