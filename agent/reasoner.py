import requests
import json
from queue import Empty
from config import OLLAMA_URL, REASON_MODEL, MAX_TOKENS_AGENT, REASON_TIMEOUT

# simple validator - ensure it's JSON parseable and has required keys
def validate_scene_summary(text):
    try:
        j = json.loads(text)
    except Exception:
        return None
    # minimal validation (you can expand)
    if not isinstance(j, dict):
        return None
    return j

def validate_decision(text):
    try:
        j = json.loads(text)
    except Exception:
        return None
    if not isinstance(j, dict):
        return None
    # required keys
    if not all(k in j for k in ("changed", "attention", "action")):
        return None
    return j

def reasoner_worker(scene_queue, stop_event):
    print("[REASONER] worker started")
    while not stop_event.is_set():
        try:
            scene_summary = scene_queue.get(timeout=0.5)
        except Empty:
            continue

        if not scene_summary or not scene_summary.strip():
            print("[REASONER] empty scene, skipping")
            continue

        print("[REASONER] input:", scene_summary[:200])
        parsed = validate_scene_summary(scene_summary)
        if parsed is None:
            print("[REASONER] invalid JSON summary, skipping")
            continue

        prompt = """
You are a monitoring agent.

Scene summary (JSON):
{scene}

Decide and respond with ONE JSON object only.

Rules:
- Use only lowercase true/false
- action must be one of: IGNORE, LOG, ALERT
- Do NOT include code fences
- Do NOT include explanations

Example of a valid response:
{{"changed": true, "attention": false, "action": "IGNORE"}}

Now produce the response.
""".format(scene=scene_summary)



        payload = {
            "model": REASON_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": MAX_TOKENS_AGENT, "temperature": 0.05}
        }

        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=REASON_TIMEOUT)
            r.raise_for_status()
            out = r.json().get("response", "").strip()
            print("[REASONER] raw output:", out)
            dec = validate_decision(out)
            if dec is None:
                print("[REASONER] invalid decision JSON, discarding")
                continue
            print("[REASONER] decision:", dec)
            # TODO: call alerting/logger/db with dec
        except Exception as e:
            print("[REASONER] API/Network error:", e)
            continue
