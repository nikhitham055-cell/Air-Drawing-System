class GestureRecognizer:

    def is_finger_up(self, landmarks, tip, pip):
        """
        Check whether a finger is raised.

        tip = fingertip landmark
        pip = middle joint landmark
        """

        return landmarks[tip][1] < landmarks[pip][1]

    def is_index_only(self, landmarks):
        """
        Returns True only when the index finger is up
        and the other three fingers are folded.
        """

        if len(landmarks) != 21:
            return False

        # Index finger
        index_up = self.is_finger_up(
            landmarks,
            8,
            6
        )

        # Middle finger
        middle_up = self.is_finger_up(
            landmarks,
            12,
            10
        )

        # Ring finger
        ring_up = self.is_finger_up(
            landmarks,
            16,
            14
        )

        # Pinky
        pinky_up = self.is_finger_up(
            landmarks,
            20,
            18
        )

        # Draw only if index is up
        # and all other fingers are down.
        return (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        )

    def is_two_fingers(self, landmarks):
        """
        Detect index + middle fingers raised.
        Used as the eraser gesture.
        """

        if len(landmarks) != 21:
            return False

        index_up = self.is_finger_up(
            landmarks,
            8,
            6
        )

        middle_up = self.is_finger_up(
            landmarks,
            12,
            10
        )

        ring_up = self.is_finger_up(
            landmarks,
            16,
            14
        )

        pinky_up = self.is_finger_up(
            landmarks,
            20,
            18
        )

        return (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        )
