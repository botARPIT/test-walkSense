import time

def init_scene_state():
    return {
        "people": {},
        "objects": {},
        "activities": [],
        "last_updated": None
    }

def update_scene_state(state, vision_json):
    for p in vision_json.get("people", []):
        state["people"][p["id"]] = p

    for o in vision_json.get("objects", []):
        state["objects"][o["id"]] = o

    if "activity" in vision_json:
        state["activities"].append({
            "activity": vision_json["activity"],
            "time": time.time()
        })

    state["last_updated"] = time.time()
    return state
