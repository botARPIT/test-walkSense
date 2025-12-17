import cv2
import time
from camera.camera_source import CameraSource

def camera_worker(frame_queue, stop_event, camera_config):
    """
    camera_config example:
    {
        "source": 1,            # Iriun usually shows as /dev/video1
        "width": 1280,
        "height": 720,
        "fps": 30
    }
    """
    print("[CAM] starting")

    cam = CameraSource(**camera_config).open()
    last_push = 0

    while not stop_event.is_set():
        frame = cam.read()
        if frame is None:
            print("[CAM] frame read failed")
            time.sleep(0.1)
            continue

        # Optional preview (debug only)
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

        now = time.time()
        if now - last_push > 2:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
            frame_queue.put(frame)
            print("[CAM] pushed frame to queue")
            last_push = now

    cam.release()
    cv2.destroyAllWindows()
    print("[CAM] stopped")
