import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self):
        base_options = mp.tasks.BaseOptions(
            model_asset_path="models/hand_landmarker.task"
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )

    def find_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = self.detector.detect(mp_image)

        landmarks = []

        if results.hand_landmarks:
            hand = results.hand_landmarks[0]

            height, width, _ = frame.shape

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                landmarks.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        return frame, landmarks