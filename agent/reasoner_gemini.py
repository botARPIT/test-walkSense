import os
import json
import time
from queue import Empty
import google.generativeai as genai

# --- configure Gemini once ---
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"


def validate_scene_summary(text: str):
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def validate_decision(text: str):
    try:
        data = json.loads(text)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    if not all(k in data for k in ("changed", "attention", "action")):
        return None

    if data["action"] not in ("IGNORE", "LOG", "ALERT"):
        return None

    if not isinstance(data["changed"], bool):
        return None

    if not isinstance(data["attention"], bool):
        return None

    return data


def reasoner_worker(scene_queue, stop_event):
    print("[REASONER-GEMINI] worker started")

    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            # "max_output_tokens": 64,
        },
    )

    while not stop_event.is_set():
        try:
            scene_summary = scene_queue.get(timeout=0.5)
        except Empty:
            continue

        parsed = validate_scene_summary(scene_summary)
        # if parsed is None:
        #     print("[REASONER-GEMINI] invalid scene JSON → skip")
        #     continue

        # # --- HARD GATE: do not reason if nothing changed ---
        # if not parsed.get("motion", True):
        #     print("[REASONER-GEMINI] no motion → IGNORE")
        #     continue

        # prompt = (
        #     "You are a monitoring agent.\n\n"
        #     "Scene summary (JSON):\n"
        #     f"{scene_summary}\n\n"
        #     "Respond ONLY with valid JSON.\n"
        #     "Keys:\n"
        #     "- changed: boolean\n"
        #     "- attention: boolean\n"
        #     "- action: one of IGNORE, LOG, ALERT\n"
        # )
        prompt = """
        Summarize this {parsed}.    
        """
        response = model.generate_content(prompt)
        print("[REASONER-GEMINI] API error:", response)
        # try:
        #     response = model.generate_content(prompt)
        #     raw = response.text.strip()
        # except Exception as e:
        #     print("[REASONER-GEMINI] API error:", e)
        #     continue

        # if not raw:
        #     print("[REASONER-GEMINI] empty response → skip")
        #     continue

        # print("[REASONER-GEMINI] raw output:", raw)

        # decision = validate_decision(raw)
        # if decision is None:
        #     print("[REASONER-GEMINI] invalid decision JSON → discard")
        #     continue

        # print("[REASONER-GEMINI] decision:", decision)

        # ---- PLACEHOLDER FOR ACTIONS ----
        # if decision["action"] == "ALERT":
        #     trigger_alert(decision)
        # elif decision["action"] == "LOG":
        #     log_event(decision)

        time.sleep(0.1)
