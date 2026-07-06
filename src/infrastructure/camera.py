import cv2
import numpy as np


class Camera:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera with index {camera_index}. "
                "Try camera_index=1."
            )

    def read(self) -> np.ndarray:
        success, frame = self.capture.read()

        if not success:
            raise RuntimeError("Could not read frame from camera.")

        return frame

    def release(self) -> None:
        self.capture.release()