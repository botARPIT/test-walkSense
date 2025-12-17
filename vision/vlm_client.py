import base64
import requests
import cv2
import time
from queue import Empty
from config import OLLAMA_URL, VISION_MODEL, MAX_TOKENS_VISION, VISION_TIMEOUT, VISION_MAX_RETRIES, JPEG_QUALITY

def encode_frame(frame):
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(jpeg).decode()

def _call_vision_api(image_b64, timeout):
#     prompt = """
# You see a single camera frame.
# Describe ONLY high-level changes (no prose). Return strictly valid JSON.

# Schema:
# {
#   "motion": true|false,
#   "objects": ["person","car",...],
#   "object_count": number
# }
# """
    prompt = "Describe what you see"
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS_VISION,
            "temperature": 0.05
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")

def vision_worker(frame_queue, scene_queue, stop_event):
    print("[VISION] worker started")
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except Empty:
            continue

        print("[VISION] frame received")
        image_b64 = encode_frame(frame)

        summary = ""
        for attempt in range(1, VISION_MAX_RETRIES + 2):  # try 1..N+1
            try:
                summary = _call_vision_api(image_b64, timeout=VISION_TIMEOUT)
                if summary and summary.strip():
                    break
                print(f"[VISION] empty summary on attempt {attempt}")
            except Exception as e:
                print("[VISION] API error attempt", attempt, e)
                time.sleep(min(2 * attempt, 5))
                continue

        if not summary or not summary.strip():
            print("[VISION] skipping: no valid summary after retries")
            continue

        summary = summary.strip()
        print("[VISION] summary:", summary)
        # push validated summary to scene_queue if space
        try:
            scene_queue.put_nowait(summary)
        except Exception as e:
            print("[VISION] scene_queue full, dropping summary:", e)
