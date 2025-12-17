import cv2
import time

class CameraSource:
    def __init__(self, source, width=None, height=None, fps=None):
        """
        source:
          - int (0,1,2...) for USB / Iriun webcam
          - str (rtsp:// or http://) for IP camera
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")

        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        return self

    def read(self):
        if not self.cap:
            raise RuntimeError("Camera not opened")

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
