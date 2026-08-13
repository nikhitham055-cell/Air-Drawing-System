#  AI Air Drawing System

An AI-powered computer vision application that allows users to **draw in the air using hand gestures** without touching a physical surface.

The system uses **MediaPipe for real-time hand tracking**, computer vision techniques for gesture recognition, and a trained machine learning model to recognize drawn shapes such as circles, squares, stars, and triangles.

##  Features

-  Real-time hand detection and tracking
-  Touch-free air drawing using hand gestures
-  Interactive virtual drawing canvas
-  Gesture-based drawing controls
-  AI-based shape recognition
-  Recognition of multiple geometric shapes
-  Real-time webcam integration
-  Real-time processing and prediction

## Project Structure

Air-Drawing-System/
│
├── models/
├── ai_recognition.py
├── drawing_canvas.py
├── gesture_recognition.py
├── hand_tracker.py
├── main.py
├── train_model.py
├── requirements.txt
└── README.md

##  Supported Shapes

The system is trained to recognize:

- Circle
- Square
- Star
- Triangle

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Computer vision and webcam processing |
| MediaPipe | Real-time hand tracking |
| NumPy | Numerical computation and data processing |
| TensorFlow | Machine learning model training |
| Scikit-learn | Model evaluation and preprocessing |

##  System Architecture

```text
Webcam
   │
   ▼
Hand Detection
   │
   ▼
Hand Landmark Tracking
   │
   ▼
Gesture Recognition
   │
   ▼
Air Drawing Canvas
   │
   ▼
Shape Data Collection
   │
   ▼
Machine Learning Model
   │
   ▼
Shape Prediction
   │
   ├── Circle
   ├── Square
   ├── Star
   └── Triangle

