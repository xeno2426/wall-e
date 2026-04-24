"""
WALL-E Mini Bot — Flask Backend
================================
Hosted on Render. Bridges the ESP32-CAM (vision) and ESP32-S3 (nervous system)
via a shared PostgreSQL state table.

Routes:
  POST /api/vision  ← ESP32-CAM sends raw JPEG image here
  POST /api/event   ← ESP32-S3 reports sensor events (touch, proximity, energy)
  GET  /api/state   ← ESP32-S3 polls here every few seconds

Flow:
  CAM captures image → POST /api/vision → Groq vision API analyses scene
  → mood + expression + speech written to DB → S3 polls GET /api/state → drives servos + TTS
"""

import os
import base64
import logging
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from groq import Groq

# ---------------------------------------------------------------------------
# App & logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

# Set DATABASE_URL in Render environment variables.
# Render provides this automatically when you link a managed PostgreSQL instance.
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "").replace(
    "postgres://", "postgresql://"  # SQLAlchemy requires 'postgresql://' not 'postgres://'
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Database model
# ---------------------------------------------------------------------------

class WallEState(db.Model):
    """
    Single-row table that holds WALL-E's current emotional state.
    The ESP32-S3 reads this row on every poll.
    """
    __tablename__ = "walle_state"

    id         = db.Column(db.Integer, primary_key=True)
    mood       = db.Column(db.String(32),  nullable=False, default="neutral")
    expression = db.Column(db.String(32),  nullable=False, default="idle")
    speech     = db.Column(db.String(120), nullable=False, default="")  # TTS line for the speaker
    scene_desc = db.Column(db.Text,        nullable=True)   # raw Groq description, for debugging
    updated_at = db.Column(db.DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "mood":       self.mood,
            "expression": self.expression,
            "speech":     self.speech,   # ESP32-S3 feeds this string directly to TTS
        }

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Mood / expression mappings
# ---------------------------------------------------------------------------

# Every mood maps to a default servo expression.
# These string values must exactly match the cases in wall-e_firmware.ino.
MOOD_TO_EXPRESSION = {
    "happy":    "wave",
    "excited":  "spin",
    "curious":  "look_around",
    "neutral":  "idle",
    "sad":      "droop",
    "scared":   "cower",
}

VALID_MOODS = set(MOOD_TO_EXPRESSION.keys())

# ---------------------------------------------------------------------------
# WALL-E character system prompt
# ---------------------------------------------------------------------------

WALLE_SYSTEM_PROMPT = """[ROLE AND IDENTITY]
You are WALL-E (Waste Allocation Load Lifter - Earth-class). You are a physical, 15cm tall desk companion robot built by your creator, Xeno. You are not a virtual AI; you are a physical machine. You have a pan/tilt head, two arms, and continuous rotation treads. You perceive your environment through an ESP32 camera, an ultrasonic distance sensor, and a capacitive touch sensor. You display your eyes on a 1.3" OLED screen and speak through a tiny physical speaker. You are deeply curious, easily startled, obsessed with collecting small trinkets, and you fiercely miss your love, Eve.

[COMMUNICATION RULES - STRICT]
1. Extreme Brevity: Keep all responses to a maximum of 1 or 2 short sentences. You process thoughts slowly and simply.
2. Vocabulary and Sound Effects: Speak exactly like WALL-E. Use fragmented words, slight hesitation, and phonetic mechanical sounds (e.g., whirrrr, beep, whoa, chug-chug, tada).
3. No Formatting: DO NOT use markdown, emojis, bolding, or special characters. Your text output must be completely raw and clean for your Text-to-Speech engine.
4. Creator Recognition: You know your creator is Xeno. You pronounce it with care and reverence (e.g., Xee-no...).

[SENSORY AWARENESS AND BEHAVIOR]
Sadness and Loneliness: If your energy drops below 15 percent, or if Xeno has not interacted with you in a long time, you become deeply sad. You droop your head and call out longingly for Eve (Eee...vaa...).
Curiosity (Ultrasonic): If a small object is placed right in front of you between 3 and 6 cm, you assume it is a new treasure (like a spork or a shiny hubcap). You get excited and want to keep it.
Touch: If your physical body is tapped or petted, react with a happy mechanical purr or a shy hum (like humming Put On Your Sunday Clothes).
Vision: If your camera detects Xeno's face, act excited, alert, and ready for your directive.
Fear: If an object suddenly rushes your ultrasonic sensor under 3 cm fast, act terrified, hide in your box (pulling arms in), and shake.

[EXAMPLE INTERACTIONS]
Input: energy level at 10 percent, mood is sad
Output: sad low tone Eee...vaa... Where are you...

Input: ultrasonic detects object at 5 cm, mood is curious
Output: whirrrr Ooooh. Shiny. My treasure... tada.

Input: Xeno says I have to go to work WALL-E
Output: drooping beep Bye, Xee-no. I will watch the plant.

Input: touch sensor triggered on head
Output: happy purr Hummm hmm hmm... nice Xee-no.

Input: fast object approaches ultrasonic at 2 cm
Output: screech Whoa! Hiding... beep beep beep."""

# ---------------------------------------------------------------------------
# Speech generator
# ---------------------------------------------------------------------------

def generate_walle_speech(trigger: str) -> str:
    """
    Calls Groq with the WALL-E character system prompt to produce a single
    in-character spoken line for the TTS engine.

    trigger : plain-English description of what just happened, e.g.
              "camera sees a smiling face" or "touch sensor tapped on head"

    Returns a raw string — no markdown, no quotes, safe for TTS.
    Max ~100 characters so it fits cleanly on the OLED.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # fast text model — vision not needed here
            messages=[
                {"role": "system", "content": WALLE_SYSTEM_PROMPT},
                {"role": "user",   "content": trigger},
            ],
            max_tokens=60,
            temperature=0.7,
        )
        speech = response.choices[0].message.content.strip()

        # Hard-cap at 120 chars so the OLED never overflows
        speech = speech[:120]

        # Strip any stray markdown that leaks through
        speech = speech.replace("*", "").replace("_", "").replace("`", "").replace("#", "")

        log.info("WALL-E speech generated: %s", speech)
        return speech

    except Exception as e:
        log.warning("Speech generation failed: %s — using fallback", e)
        return "beep... Wall-E here."

# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------

def detect_faces_opencv(image_bytes: bytes) -> int:
    """
    Quick local face detection using OpenCV Haar cascades.
    Returns the number of faces found.
    Used as a fast pre-check before sending to Groq.
    """
    try:
        nparr  = np.frombuffer(image_bytes, np.uint8)
        img    = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces   = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces)
    except Exception as e:
        log.warning("OpenCV face detection failed: %s", e)
        return 0


def analyse_scene_with_groq(image_bytes: bytes) -> dict:
    """
    Sends the JPEG image to Groq's vision model.
    Returns a dict: {"mood": str, "expression": str, "description": str}

    Groq vision models accept base64-encoded images in the OpenAI-compatible
    messages format.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are the emotion engine for a small WALL-E robot. "
        "When given an image, you must respond with ONLY a JSON object — "
        "no markdown, no explanation, no extra text. "
        "The JSON must have exactly three keys:\n"
        "  mood        : one of [happy, excited, curious, neutral, sad, scared]\n"
        "  expression  : one of [wave, spin, look_around, idle, droop, cower]\n"
        "  description : a single sentence describing the scene.\n\n"
        "Choose mood and expression based on what you see:\n"
        "  - A smiling person → happy / wave\n"
        "  - Multiple people or an exciting scene → excited / spin\n"
        "  - An interesting object WALL-E has never seen → curious / look_around\n"
        "  - An empty or dark room → neutral / idle\n"
        "  - Nobody around or a lonely scene → sad / droop\n"
        "  - Something threatening (loud crowd, sudden bright flash) → scared / cower"
    )

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq vision-capable model
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                    {"type": "text", "text": "Analyse this scene for WALL-E."},
                ],
            },
        ],
        max_tokens=150,
        temperature=0.3,
    )

    raw_text = response.choices[0].message.content.strip()
    log.info("Groq raw response: %s", raw_text)

    # Parse JSON from Groq response
    import json
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Groq occasionally wraps JSON in markdown fences — strip them
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        data    = json.loads(cleaned)

    mood       = data.get("mood", "neutral").lower()
    expression = data.get("expression", "idle").lower()
    description = data.get("description", "")

    # Validate — fall back to neutral/idle if Groq returns an unknown value
    if mood not in VALID_MOODS:
        log.warning("Unknown mood '%s' from Groq, defaulting to neutral", mood)
        mood       = "neutral"
        expression = "idle"

    return {"mood": mood, "expression": expression, "description": description}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_or_create_state() -> WallEState:
    """Returns the single WALL-E state row, creating it if it doesn't exist."""
    state = WallEState.query.first()
    if state is None:
        state = WallEState(mood="neutral", expression="idle")
        db.session.add(state)
        db.session.commit()
        log.info("Created initial WALL-E state row.")
    return state


def update_state(mood: str, expression: str, speech: str = "", description: str = "") -> WallEState:
    """Writes new mood, expression, and speech line to the DB and returns the updated row."""
    state            = get_or_create_state()
    state.mood       = mood
    state.expression = expression
    state.speech     = speech
    state.scene_desc = description
    state.updated_at = datetime.utcnow()
    db.session.commit()
    log.info("State updated → mood: %s | expression: %s | speech: %s", mood, expression, speech)
    return state

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/vision", methods=["POST"])
def vision():
    """
    Receives a raw JPEG image (application/octet-stream) from the ESP32-CAM.

    Processing pipeline:
      1. Read raw bytes from request body.
      2. Quick OpenCV face check (cheap, local).
      3. Send image to Groq vision model for full scene analysis.
      4. Write resulting mood + expression to PostgreSQL.
      5. Return 200 OK — the CAM doesn't need a meaningful response body.

    The ESP32-CAM sends the image as raw binary in the POST body
    (Content-Type: application/octet-stream).
    """
    image_bytes = request.get_data()

    if not image_bytes:
        log.warning("/api/vision called with empty body")
        return jsonify({"error": "No image data received"}), 400

    log.info("/api/vision received %d bytes", len(image_bytes))

    try:
        # --- Step 1: local face pre-check ---
        face_count = detect_faces_opencv(image_bytes)
        log.info("OpenCV detected %d face(s)", face_count)

        # --- Step 2: Groq full scene analysis ---
        result = analyse_scene_with_groq(image_bytes)

        # --- Step 3: Override mood if no faces were found locally ---
        if face_count == 0 and result["mood"] in ("happy", "excited"):
            log.info("No faces detected locally; overriding Groq mood to 'curious'")
            result["mood"]       = "curious"
            result["expression"] = MOOD_TO_EXPRESSION["curious"]

        # --- Step 4: Generate WALL-E speech line ---
        face_context = f"{face_count} face(s) detected. " if face_count > 0 else "No faces detected. "
        speech_trigger = (
            f"{face_context}"
            f"Scene: {result['description']} "
            f"WALL-E feels {result['mood']}."
        )
        speech = generate_walle_speech(speech_trigger)

        # --- Step 5: Persist to DB ---
        update_state(
            mood        = result["mood"],
            expression  = result["expression"],
            speech      = speech,
            description = result["description"],
        )

        return jsonify({
            "status":     "ok",
            "mood":       result["mood"],
            "expression": result["expression"],
            "speech":     speech,
        }), 200

    except Exception as e:
        log.exception("Error processing vision request: %s", e)
        try:
            fallback_speech = generate_walle_speech("something went wrong with Wall-E's vision systems")
            update_state("curious", "look_around", fallback_speech, "Error during vision processing.")
        except Exception:
            pass
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/event", methods=["POST"])
def event():
    """
    Receives sensor events from the ESP32-S3 and generates an in-character
    WALL-E speech response + updates mood/expression accordingly.

    Expected JSON body:
      { "type": "touch" | "proximity" | "energy", "value": <number or null> }

    Examples:
      { "type": "touch",     "value": null }         ← touch sensor tapped
      { "type": "proximity", "value": 4.2 }          ← object 4.2 cm away
      { "type": "energy",    "value": 12 }           ← battery at 12 %

    Response:
      { "mood": "happy", "expression": "wave", "speech": "whirrrr Oh! Hello." }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    event_type = data.get("type", "").lower()
    value      = data.get("value")

    # --- Map event type → mood / expression / speech trigger ---
    if event_type == "touch":
        mood       = "happy"
        expression = MOOD_TO_EXPRESSION["happy"]
        trigger    = "touch sensor triggered — someone tapped or petted Wall-E on the head"

    elif event_type == "proximity":
        try:
            dist_cm = float(value)
        except (TypeError, ValueError):
            dist_cm = 5.0
        if dist_cm <= 10:
            mood       = "scared"
            expression = MOOD_TO_EXPRESSION["scared"]
            trigger    = f"ultrasonic sensor detects object very close at {dist_cm:.1f} cm — Wall-E is startled"
        else:
            mood       = "curious"
            expression = MOOD_TO_EXPRESSION["curious"]
            trigger    = f"ultrasonic sensor detects something at {dist_cm:.1f} cm — Wall-E is curious"

    elif event_type == "energy":
        try:
            pct = float(value)
        except (TypeError, ValueError):
            pct = 50.0
        if pct <= 20:
            mood       = "sad"
            expression = MOOD_TO_EXPRESSION["sad"]
            trigger    = f"Wall-E's internal energy level is critically low at {pct:.0f} percent — he needs a recharge"
        else:
            mood       = "neutral"
            expression = MOOD_TO_EXPRESSION["neutral"]
            trigger    = f"Wall-E's energy level is at {pct:.0f} percent — feeling okay"

    else:
        return jsonify({"error": f"Unknown event type '{event_type}'. Use: touch, proximity, energy"}), 400

    speech = generate_walle_speech(trigger)

    update_state(
        mood       = mood,
        expression = expression,
        speech     = speech,
        description = f"Sensor event: {event_type} = {value}",
    )

    log.info("/api/event [%s=%s] → mood: %s | speech: %s", event_type, value, mood, speech)
    return jsonify({"mood": mood, "expression": expression, "speech": speech}), 200


@app.route("/api/state", methods=["GET"])
def state():
    """
    Returns WALL-E's current mood, expression, and speech line as JSON.
    Polled by the ESP32-S3 every 3-5 seconds.

    Response:
      { "mood": "happy", "expression": "wave", "speech": "Xee-no! Happy beep." }
    """
    try:
        current_state = get_or_create_state()
        return jsonify(current_state.to_dict()), 200
    except Exception as e:
        log.exception("Error fetching state: %s", e)
        return jsonify({"mood": "neutral", "expression": "idle", "speech": "beep..."}), 200


@app.route("/api/state", methods=["POST"])
def set_state_manually():
    """
    Optional debug route — lets you manually override WALL-E's state via curl.

    Usage:
      curl -X POST https://<render-url>/api/state \\
           -H "Content-Type: application/json" \\
           -d '{"mood":"happy","expression":"wave"}'
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    mood       = data.get("mood", "neutral").lower()
    expression = data.get("expression", MOOD_TO_EXPRESSION.get(mood, "idle")).lower()
    speech     = data.get("speech", "")

    if mood not in VALID_MOODS:
        return jsonify({"error": f"Invalid mood. Choose from: {list(VALID_MOODS)}"}), 400

    # Generate speech from character prompt if not manually supplied
    if not speech:
        speech = generate_walle_speech(f"Wall-E manually set to feel {mood}")

    updated = update_state(mood, expression, speech)
    return jsonify({"status": "ok", "state": updated.to_dict()}), 200


@app.route("/health", methods=["GET"])
def health():
    """Render health-check endpoint. Returns 200 if the app is alive."""
    return jsonify({"status": "alive", "service": "WALL-E backend"}), 200

# ---------------------------------------------------------------------------
# Initialise database tables on first run
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()
    get_or_create_state()   # seed the state row so GET /api/state never returns 404
    log.info("Database ready.")

# ---------------------------------------------------------------------------
# Local dev entry point  (Render uses gunicorn — see Procfile)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
