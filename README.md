#  AI Air Drawing System

An AI-powered computer vision project that allows users to **draw in the air using hand gestures** without touching a physical surface.

The system uses **real-time hand tracking, gesture recognition, computer vision, and machine learning** to track the user's finger movements and convert them into drawings on a virtual canvas. It also recognizes geometric shapes such as circles, squares, stars, and triangles.

---

##  Project Overview

Traditional drawing applications require a physical input device such as a mouse, keyboard, or touchscreen.

The **AI Air Drawing System** provides a touch-free alternative by using a webcam to detect hand movements. The user's finger acts as a virtual drawing tool, allowing them to create shapes in the air.

The captured drawing can then be processed by an AI model to identify the drawn shape.

---

##  Features

-  Real-time hand detection and tracking
-  Touch-free air drawing
-  Virtual drawing canvas
-  AI-based shape recognition
-  Hand gesture recognition
-  Webcam-based interaction
-  Real-time computer vision processing
-  Recognition of multiple geometric shapes
-  Machine learning-based prediction

---

##  Supported Shapes

The current system supports recognition of:

- Circle
- Square
- Star
- Triangle

The system can be extended to support additional shapes by adding more training data.

---

##  Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe**
- **TensorFlow**
- **NumPy**
- **Scikit-learn**

### Computer Vision

- Hand detection
- Hand landmark tracking
- Finger tracking
- Gesture recognition
- Real-time webcam processing

### Machine Learning

- Shape classification
- Model training
- Feature extraction
- Prediction

---

## System Architecture

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
             Finger Tracking
                    │
                    ▼
            Virtual Canvas
                    │
                    ▼
             Drawing Data
                    │
                    ▼
          Machine Learning Model
                    │
                    ▼
            Shape Prediction
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Circle       Square       Star
                    │
                    ▼
                 Triangle
```
## Project Structure

```text
Air-Drawing-System/
│
├── models/
│   └── Trained model files
│
├── ai_recognition.py
├── download_circle.py
├── drawing_canvas.py
├── gesture_recognition.py
├── hand_tracker.py
├── main.py
├── train_model.py
├── requirements.txt
└── README.md
