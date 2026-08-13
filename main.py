import cv2

from hand_tracker import HandTracker
from drawing_canvas import DrawingCanvas
from gesture_recognition import GestureRecognizer
from ai_recognition import AIRecognizer

def main():

    # Start webcam
    cap = cv2.VideoCapture(0)

    # Create objects
    tracker = HandTracker()
    canvas = DrawingCanvas(640, 480)
    gesture = GestureRecognizer()
    recognizer = AIRecognizer("models/drawing_model.keras")

    while True:

        # Read webcam frame
        success, frame = cap.read()

        if not success:
            print("Could not access the webcam.")
            break

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        # Detect hand
        frame, landmarks = tracker.find_hands(frame)

        # Check if a complete hand was detected

        if len(landmarks) == 21:

            # ✌️ Eraser
            if gesture.is_two_fingers(landmarks):

                eraser_point = landmarks[8]

                canvas.erase(
                    eraser_point,
                    size=30
                )

                canvas.previous_point = None

            # ☝️ Drawing
            elif gesture.is_index_only(landmarks):

                index_finger = landmarks[8]

                canvas.draw(index_finger)

            # Other gestures = stop drawing
            else:

                canvas.draw(None)

        else:

            canvas.draw(None)

        # Get drawing
        drawing = canvas.get_canvas()

        # Combine camera and drawing
        output = cv2.addWeighted(
            frame,
            1,
            drawing,
            1,
            0
        )

        # Display
        cv2.imshow("AI Air Drawing", output)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        # Clear
        if key == ord("c"):
            canvas.clear()

        # Save
        elif key == ord("s"):
            cv2.imwrite(
                "drawings/my_drawing.png",
                drawing
            )
            print("Drawing saved!")

        # Blue
        elif key == ord("1"):
            canvas.set_color((255, 0, 0))
            print("Blue selected")

        # Red
        elif key == ord("2"):
            canvas.set_color((0, 0, 255))
            print("Red selected")

        # Green
        elif key == ord("3"):
            canvas.set_color((0, 255, 0))
            print("Green selected")

        # White
        elif key == ord("4"):
            canvas.set_color((255, 255, 255))
            print("White selected")

        # Increase brush
        elif key == ord("+") or key == ord("="):
            canvas.increase_brush()

        # Decrease brush
        elif key == ord("-"):
            canvas.decrease_brush()

        # AI prediction
        elif key == ord("p"):

            label, confidence = recognizer.predict(
                canvas.get_canvas()
            )

            if label is not None:

                print(
                    f"Prediction: {label} "
                    f"({confidence * 100:.2f}%)"
                )

            else:

                print("No drawing detected.")

        # Quit
        elif key == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
