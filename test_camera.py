from camera.capture import Camera
import cv2

cam = Camera(index=1)

while True:
    frame = cam.get_frame()
    if frame is None:
        continue

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
