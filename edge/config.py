import os

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = SCRIPT_DIR
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker.task")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

# --- PERFORMANCE CONFIG ---
CAMERA_INDEX = 0
TARGET_LAPTOP_FPS = 25
TARGET_RPI_FPS = 15
RESOLUTION_W = 640
RESOLUTION_H = 480
YOLO_FRAME_SKIP = 5  # Run YOLO every N frames to save compute

# --- THRESHOLDS ---
EAR_THRESHOLD = 0.20             # Lowered to 0.20 to eliminate false positives
PITCH_THRESHOLD = 0.35           # Pitch (up/down) limit (roughly 20 degrees)
YAW_THRESHOLD = 0.35             # Yaw (left/right) limit (roughly 20 degrees)
DISTANCE_THRESHOLD_CALLING = 0.15 # Normalized distance between wrist and ear
PHONE_PADDING = 30               # Padding (px) around phone bbox for wrist intersection

# --- EVENT DEFS & PRIORITIES ---
# Lower number means higher priority. Medical Emergency > Drowsiness > Phone
EVENT_PRIORITIES = {
    "MEDICAL_EMERGENCY": 0,
    "DROWSINESS": 1,
    "PHONE_USAGE": 2,
    "DISTRACTION": 3,
    "TEXTING": 4,           # Texting is specific distraction
    "RUBBERNECKING": 5,
    "POSTURE_COLLAPSE": 6
}

# --- EVENT TIME THRESHOLDS (Seconds) ---
# How long must a condition hold to trigger the event state?
TIME_THRESHOLDS = {
    "DROWSINESS": 1.5,   # Reduced from 3.0 so testing eye-closure is faster
    "PHONE_USAGE": 1.0,  # Reduced a bit for responsiveness
    "DISTRACTION": 3.0,
    "TEXTING": 1.5
}

# --- COOLDOWNS (Seconds) ---
# How long in seconds before the SAME event can trigger another alert?
COOLDOWNS = {
    "MEDICAL_EMERGENCY": 0,  # Never cooldown
    "DROWSINESS": 5,
    "PHONE_USAGE": 3,
    "DISTRACTION": 4,
    "TEXTING": 4,
    "RUBBERNECKING": 4,
    "POSTURE_COLLAPSE": 0
}

# --- LOGGING ---
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
CSV_LOG_FILE = os.path.join(LOG_DIR, "events.csv")
JSON_LOG_FILE = os.path.join(LOG_DIR, "events.json")
