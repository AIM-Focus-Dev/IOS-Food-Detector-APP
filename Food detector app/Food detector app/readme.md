# Food Detector App for iOS

This is an iOS app that uses machine learning to identify food items through the camera. The project is structured as follows:

## Files and Directories

### Core Files

#### `CameraProcessor.swift`

This file contains the main logic for capturing video frames using the device's camera and processing them for machine learning predictions. It also manages the camera preview layer.

#### `ContentView.swift`

This is the main SwiftUI file that defines the app's user interface. It initialises the `CameraProcessor` class and manages the state for scanning and displaying predictions.

#### `Food_detector_appApp.swift`

This is the entry point for the SwiftUI app. It initializes the `ContentView`.

#### `CameraProcessorViewModel.swift`

This ViewModel is used for data-binding between `ContentView` and `CameraProcessor`. It handles updating the UI based on predictions.

### Machine Learning

#### `ML_model`

This directory contains the Core ML model files used for food identification.

#### `FoodCL.mlpackage`

This is the machine learning model package used for predicting the food items.

#### `labels.txt`

Contains the labels for the food items that the machine learning model can identify.

#### `prediction.swift`

Contains utility functions for making predictions using the machine learning model.

### Image Preprocessing

#### `ImageFramesPreprocessing.swift`

This file contains functions for resizing and preprocessing the captured frames before feeding them into the machine learning model.

### Helper Files

#### `ProcessingHelperFiles`

Contains additional helper files and extensions used throughout the app.
