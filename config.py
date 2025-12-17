OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
VISION_MODEL = "qwen3-vl:2b"
REASON_MODEL = "gemma3:270m"

CAMERA_ID = 1        # Iriun usually
VISION_INTERVAL = 2 # seconds
MOTION_THRESHOLD = 12

JPEG_QUALITY = 80

CAMERA_ID = 1
VISION_INTERVAL = 2.0  # seconds between sending frames to vision
JPEG_QUALITY = 80

CAMERA_CONFIG = {
    "source": 0,        # Iriun usually appears as /dev/video1
    "width": 1280,
    "height": 720,
    "fps": 30
}

# timeouts (seconds)
VISION_TIMEOUT = 60
REASON_TIMEOUT = 15

# retry policy for empty vision summaries
VISION_MAX_RETRIES = 2

MAX_TOKENS_VISION = 5000
MAX_TOKENS_AGENT = 150
