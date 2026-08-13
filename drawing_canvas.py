import cv2
import numpy as np
import math


class DrawingCanvas:

    def __init__(self, width, height):

        self.canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        self.previous_point = None
        self.smoothed_point = None

        self.smoothing = 0.15
        self.dead_zone = 3

        # Drawing settings
        self.color = (255, 0, 0)
        self.brush_size = 4

    def smooth_point(self, point):

        if self.smoothed_point is None:
            self.smoothed_point = point
            return point

        old_x, old_y = self.smoothed_point
        new_x, new_y = point

        smooth_x = int(
            old_x +
            self.smoothing *
            (new_x - old_x)
        )

        smooth_y = int(
            old_y +
            self.smoothing *
            (new_y - old_y)
        )

        distance = math.sqrt(
            (smooth_x - old_x) ** 2 +
            (smooth_y - old_y) ** 2
        )

        if distance < self.dead_zone:
            return self.smoothed_point

        self.smoothed_point = (
            smooth_x,
            smooth_y
        )

        return self.smoothed_point

    def draw(self, point):

        if point is None:
            self.previous_point = None
            return

        point = self.smooth_point(point)

        if self.previous_point is not None:

            cv2.line(
                self.canvas,
                self.previous_point,
                point,
                self.color,
                self.brush_size,
                cv2.LINE_AA
            )

        self.previous_point = point

    def erase(self, point, size=30):

        if point is None:
            return

        cv2.circle(
            self.canvas,
            point,
            size,
            (0, 0, 0),
            -1
        )

    def clear(self):

        self.canvas[:] = 0

        self.previous_point = None
        self.smoothed_point = None

    def set_color(self, color):

        self.color = color

    def increase_brush(self):

        self.brush_size += 1

        if self.brush_size > 20:
            self.brush_size = 20

    def decrease_brush(self):

        self.brush_size -= 1

        if self.brush_size < 1:
            self.brush_size = 1

    def get_canvas(self):

        return self.canvas
