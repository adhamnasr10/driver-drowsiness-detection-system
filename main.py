# =============================================================================
# main.py — Driver Drowsiness Detection System  (entry point)
# =============================================================================
# Controls the main video loop:
#   1. Capture frame from webcam
#   2. Pass to DrowsinessDetector.process()
#   3. Show annotated frame in a window
#   4. If drowsy → trigger alarm
#   5. Exit cleanly on ESC key or window close
# =============================================================================

import sys
import cv2

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
from detector import DrowsinessDetector
from alarm import trigger_alarm, reset_alarm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Open the webcam and configure resolution.

    Raises SystemExit with a friendly message if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"[ERROR] Cannot open camera at index {index}.\n"
            "  → Check that your webcam is connected and not in use by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"[INFO] Camera opened  (index={index}, {width}×{height})")
    return cap


def print_startup_banner() -> None:
    """Print a short startup summary to the console."""
    from config import EAR_THRESHOLD, EAR_CONSEC_FRAMES, MODEL_PATH
    print("=" * 60)
    print("  Driver Drowsiness Detection System")
    print("  Powered by MediaPipe Face Landmarker + EAR")
    print("=" * 60)
    print(f"  Model       : {MODEL_PATH}")
    print(f"  EAR thresh  : {EAR_THRESHOLD}")
    print(f"  Consec frames: {EAR_CONSEC_FRAMES}")
    print("  Press  ESC  to quit")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    print_startup_banner()

    # -- Initialise detector (loads the MediaPipe model) ----------------------
    print("[INFO] Loading Face Landmarker model …")
    try:
        detector = DrowsinessDetector()
    except Exception as exc:
        sys.exit(
            f"[ERROR] Failed to load model: {exc}\n"
            "  → Make sure 'face_landmarker.task' is in the project folder.\n"
            "  → See README / instructions for the download command."
        )
    print("[INFO] Model loaded successfully.")

    # -- Open webcam ----------------------------------------------------------
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    window_name = "Driver Drowsiness Detection  (ESC to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # -- Video loop -----------------------------------------------------------
    while True:
        ret, frame = cap.read()

        # Camera disconnected mid-session
        if not ret or frame is None:
            print("[WARNING] Failed to read frame — retrying …")
            continue

        # Flip horizontally for a mirror-view (more natural for driver-facing cam)
        frame = cv2.flip(frame, 1)

        # ---- Run detection --------------------------------------------------
        annotated_frame, ear, is_drowsy = detector.process(frame)

        # ---- Audio alert ----------------------------------------------------
        if is_drowsy:
            trigger_alarm()     # No-op if beep already running
        else:
            reset_alarm()       # Allow alarm to fire again after driver recovers

        # ---- Display --------------------------------------------------------
        cv2.imshow(window_name, annotated_frame)

        # ---- Console feedback (optional, useful for debugging) --------------
        status = "DROWSY" if is_drowsy else "awake"
        print(f"\r  EAR={ear:.3f}  status={status}      ", end="", flush=True)

        # ---- Key handling ---------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC
            print("\n[INFO] ESC pressed — exiting.")
            break

        # Also exit if the user closes the window with the × button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\n[INFO] Window closed — exiting.")
            break

    # -- Cleanup --------------------------------------------------------------
    print("[INFO] Releasing resources …")
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
