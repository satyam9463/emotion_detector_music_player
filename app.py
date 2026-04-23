"""
MoodWave CV  —  facial emotion → music
Press SCAN button → one snapshot → DeepFace → playlist
"""

import base64, json, os, re, random
import numpy as np
import cv2
from flask import Flask, render_template, request, Response

app = Flask(__name__)

# ── Custom JSON encoder: converts ALL numpy types to plain Python ─────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):   return float(obj)
        if isinstance(obj, np.integer):    return int(obj)
        if isinstance(obj, np.ndarray):    return obj.tolist()
        return super().default(obj)

def safe_json(data, status=200):
    return Response(
        json.dumps(data, cls=SafeEncoder),
        status=status,
        mimetype="application/json"
    )

# ── DeepFace ──────────────────────────────────────────────────────────────────
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
    print("✅ DeepFace ready")
except Exception:
    HAS_DEEPFACE = False
    print("⚠️  DeepFace not found — demo mode")

# ── OpenCV face detector ──────────────────────────────────────────────────────
try:
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    FACE_CASCADE  = cv2.CascadeClassifier(_cascade_path)
    HAS_CV = True
    print("✅ OpenCV ready")
except Exception:
    HAS_CV = False
    print("⚠️  OpenCV not found")

# ── Mappings ──────────────────────────────────────────────────────────────────
EMOTION_MOOD = {
    "happy":"happy", "surprise":"energetic", "neutral":"calm",
    "sad":"sad",     "angry":"angry",        "disgust":"angry", "fear":"sad",
}

MOOD_META = {
    "happy":     {"emoji":"😄","color":"#FFD700","bg":"#100e00","label":"Joyful & Bright"},
    "sad":       {"emoji":"😢","color":"#6B8CFF","bg":"#04060f","label":"Melancholic & Deep"},
    "angry":     {"emoji":"😠","color":"#FF4444","bg":"#100000","label":"Intense & Raw"},
    "calm":      {"emoji":"😌","color":"#7FFFD4","bg":"#010f0c","label":"Peaceful & Still"},
    "energetic": {"emoji":"⚡","color":"#FF8C00","bg":"#100600","label":"Pumped & Alive"},
    "romantic":  {"emoji":"🥰","color":"#FF69B4","bg":"#100008","label":"Warm & Tender"},
    "focused":   {"emoji":"🎯","color":"#00BFFF","bg":"#00080f","label":"Sharp & Clear"},
}

# ── Load playlists ────────────────────────────────────────────────────────────
def yt_id(s):
    """Extract YouTube video ID from URL or return as-is."""
    s = s or ""
    if re.match(r'^[A-Za-z0-9_-]{11}$', s):
        return s
    for pat in [r'[?&]v=([A-Za-z0-9_-]{11})',
                r'youtu\.be/([A-Za-z0-9_-]{11})',
                r'embed/([A-Za-z0-9_-]{11})']:
        m = re.search(pat, s)
        if m: return m.group(1)
    return s

def load_playlists():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playlists.json")
    if not os.path.exists(path):
        print("⚠️  playlists.json not found")
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for mood, tracks in raw.items():
        out[mood] = [
            {
                "title":    t.get("title","?"),
                "artist":   t.get("artist",""),
                "yt_id":    yt_id(t.get("url","") or t.get("yt_id","")),
                "duration": t.get("duration","—"),
                "cover":    t.get("cover","🎵"),
            }
            for t in tracks
        ]
    total = sum(len(v) for v in out.values())
    print(f"✅ Playlists: {total} tracks across {len(out)} moods")
    return out

PLAYLISTS = load_playlists()

# ── Image decode ──────────────────────────────────────────────────────────────
def decode_frame(data_url: str):
    """base64 data URL → OpenCV BGR image."""
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# ── Detection ─────────────────────────────────────────────────────────────────
def detect(img):
    """
    Run emotion detection.
    Returns dict  OR  None if no face found.
    All numeric values are plain Python float/int (no numpy types).
    """
    if HAS_DEEPFACE:
        try:
            raw = DeepFace.analyze(img, actions=["emotion"],
                                   enforce_detection=False, silent=True)
            if isinstance(raw, list):
                raw = raw[0]

            # ── Force all values to plain Python types ──
            emotions  = {k: float(v) for k, v in raw["emotion"].items()}
            dominant  = str(raw["dominant_emotion"])
            region    = {k: int(v) for k, v in raw.get("region", {}).items()
                         if k in ("x","y","w","h")}

            return {
                "dominant": dominant,
                "emotions": dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)),
                "region":   region,
                "conf":     round(emotions.get(dominant, 0.0), 1),
                "demo":     False,
                "engine":   "DeepFace",
            }
        except Exception as e:
            print(f"[DeepFace error] {e}")
            # fall through to OpenCV demo

    # ── OpenCV demo fallback ──
    if not HAS_CV:
        return None
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return None

    bright = float(np.mean(gray)) / 255.0
    contr  = float(np.std(gray))  / 128.0
    if   bright > 0.65: dom = "happy"
    elif bright < 0.30: dom = "sad"
    elif contr  > 0.60: dom = "surprise"
    else:               dom = "neutral"

    emo = {k: 0.0 for k in ["happy","sad","neutral","angry","surprise","fear","disgust"]}
    emo[dom] = round(random.uniform(60, 82), 1)
    rest = 100.0 - emo[dom]
    others = [k for k in emo if k != dom]
    for i, k in enumerate(others):
        emo[k] = round(rest / (len(others) + i * 0.5), 1)

    x, y, w, h = [int(v) for v in faces[0]]
    return {
        "dominant": dom,
        "emotions": dict(sorted(emo.items(), key=lambda x: x[1], reverse=True)),
        "region":   {"x":x,"y":y,"w":w,"h":h},
        "conf":     emo[dom],
        "demo":     True,
        "engine":   "OpenCV Demo",
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return safe_json({
        "deepface": HAS_DEEPFACE,
        "opencv":   HAS_CV,
        "tracks":   sum(len(v) for v in PLAYLISTS.values()),
    })

@app.route("/api/scan", methods=["POST"])
def scan():
    try:
        body = request.get_json(force=True, silent=True) or {}
        img_data = body.get("image","")
        if not img_data:
            return safe_json({"ok": False, "error": "No image sent"}, 400)

        img = decode_frame(img_data)
        if img is None or img.size == 0:
            return safe_json({"ok": False, "error": "Could not decode image"}, 400)

        result = detect(img)

        if result is None:
            return safe_json({"ok": True, "face": False,
                              "msg": "No face detected — face the camera and ensure good lighting"})

        emotion  = result["dominant"]
        mood     = EMOTION_MOOD.get(emotion, "calm")
        meta     = MOOD_META.get(mood, MOOD_META["calm"])
        playlist = PLAYLISTS.get(mood, [])

        print(f"[SCAN] {result['engine']}: {emotion} → {mood}  ({result['conf']:.1f}%)")

        return safe_json({
            "ok":       True,
            "face":     True,
            "emotion":  emotion,
            "emotions": result["emotions"],
            "conf":     result["conf"],
            "region":   result["region"],
            "mood":     mood,
            "meta":     meta,
            "playlist": playlist,
            "demo":     result["demo"],
            "engine":   result["engine"],
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[FATAL]\n{tb}")
        return safe_json({"ok": False, "error": str(e)}, 500)


if __name__ == "__main__":
    print(f"\n🎵 MoodWave CV")
    print(f"   http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
