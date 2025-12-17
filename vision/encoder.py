import cv2
import base64

def encode_frame(frame, quality=80):
    _, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return base64.b64encode(buffer).decode("utf-8")
