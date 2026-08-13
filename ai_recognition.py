import cv2
import numpy as np
import tensorflow as tf


class AIRecognizer:

    def __init__(self, model_path):

        self.model = tf.keras.models.load_model(
            model_path
        )

        self.classes = [
            "CIRCLE",
            "SQUARE",
            "TRIANGLE",
            "STAR"
        ]

    def preprocess(self, canvas):

        # Convert to grayscale
        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )

        # Find drawing pixels
        coords = cv2.findNonZero(gray)

        if coords is None:
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(
            coords
        )

        # Crop drawing
        cropped = gray[
            y:y + h,
            x:x + w
        ]

        # Make square canvas
        size = max(w, h)

        square = np.zeros(
            (size, size),
            dtype=np.uint8
        )

        # Center drawing
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2

        square[
            offset_y:offset_y + h,
            offset_x:offset_x + w
        ] = cropped

        # Resize to 28x28
        resized = cv2.resize(
            square,
            (28, 28)
        )

        # Normalize
        resized = resized.astype(
            "float32"
        ) / 255.0

        # Add dimensions
        resized = resized.reshape(
            1,
            28,
            28,
            1
        )

        return resized

    def predict(self, canvas):

        processed = self.preprocess(
            canvas
        )

        if processed is None:
            return None, 0.0

        predictions = self.model.predict(
            processed,
            verbose=0
        )[0]

        index = np.argmax(
            predictions
        )

        confidence = predictions[index]

        label = self.classes[index]

        return label, confidence
