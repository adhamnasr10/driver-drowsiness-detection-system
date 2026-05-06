# =============================================================================
# config.py — Central configuration for the Drowsiness Detection System
# =============================================================================
# All tunable parameters live here so you never need to hunt through code.

# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR) settings
# ---------------------------------------------------------------------------
# EAR drops sharply when the eye closes.  A value below EAR_THRESHOLD for
# EAR_CONSEC_FRAMES consecutive frames is considered a drowsy event.
EAR_THRESHOLD: float = 0.25          # Eyes considered "closed" below this value
EAR_CONSEC_FRAMES: int = 20          # Consecutive frames before alert fires

# ---------------------------------------------------------------------------
# MediaPipe model
# ---------------------------------------------------------------------------
# Download the model with the command in README / instructions below.
# Place face_landmarker.task in the same folder as this file (or update path).
MODEL_PATH: str = "face_landmarker.task"

# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0                # 0 = default/built-in webcam
FRAME_WIDTH: int = 1080
FRAME_HEIGHT: int = 1920

# ---------------------------------------------------------------------------
# Display colours  (BGR format for OpenCV)
# ---------------------------------------------------------------------------
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)

# ---------------------------------------------------------------------------
# Alarm (winsound.Beep)
# ---------------------------------------------------------------------------
ALARM_FREQ_HZ: int  = 1000           # Beep frequency in Hertz
ALARM_DURATION_MS: int = 200         # Beep duration in milliseconds
