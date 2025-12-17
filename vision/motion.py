import cv2

def should_run_vision(last_frame, frame, threshold):
    if last_frame is None:
        return True
    diff = cv2.absdiff(last_frame, frame)
    return diff.mean() > threshold
