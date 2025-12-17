from threading import Thread, Event
from queue import Queue
from camera.capture import camera_worker
from vision.vlm_client import vision_worker
from agent.reasoner_gemini import reasoner_worker
from config import CAMERA_CONFIG

frame_queue = Queue(maxsize=1)
scene_queue = Queue(maxsize=1)
stop_event = Event()

threads = [
    Thread(target=camera_worker, args=(frame_queue, stop_event, CAMERA_CONFIG)),
    Thread(target=vision_worker, args=(frame_queue, scene_queue, stop_event)),
    Thread(target=reasoner_worker, args=(scene_queue, stop_event)),
]

for t in threads:
    t.daemon = True
    t.start()

try:
    threads[0].join()
except KeyboardInterrupt:
    stop_event.set()
    print("Shutting down...")

