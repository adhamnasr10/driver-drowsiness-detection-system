# =============================================================================
# detector.py — Face Landmarker + EAR-based drowsiness state machine
# =============================================================================
# Uses the MediaPipe Tasks API (mp.tasks.*) — NOT the deprecated mp.solutions.
#
# Eye landmark index reference (MediaPipe 478-point face mesh):
#   Left  eye: [33, 160, 158, 133, 153, 144]
#   Right eye: [362, 385, 387, 263, 373, 380]
#
# Each group of 6 indices maps to the standard EAR layout:
#   [0]=corner-left, [1]=upper-left, [2]=upper-right,
#   [3]=corner-right, [4]=lower-right, [5]=lower-left
# =============================================================================

import cv2
import numpy as np
from typing import Tuple, Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import (
    MODEL_PATH,
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
)
from utils import (
    compute_ear,
    draw_eye_landmarks,
    landmark_to_pixel,
    put_text_with_background,
)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices for each eye (6 points each)
# ---------------------------------------------------------------------------
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


class DrowsinessDetector:
    """
    Wraps the MediaPipe Face Landmarker and implements the EAR-based
    drowsiness detection state machine.

    Usage
    -----
        detector = DrowsinessDetector()
        annotated_frame, ear, is_drowsy = detector.process(bgr_frame)
    """

    def __init__(self) -> None:
        """Initialise the Face Landmarker from the .task model file."""

        # -- Build FaceLandmarkerOptions -----------------------------------------
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,       # Not needed; saves compute
            output_facial_transformation_matrixes=False,
            num_faces=1,                          # We only care about the driver
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # -- Drowsiness state -------------------------------------------------------
        self._frame_counter: int = 0    # Consecutive frames with EAR < threshold
        self._is_drowsy: bool = False   # Current drowsy flag

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def process(
        self,
        bgr_frame: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Detect face landmarks, compute EAR, update drowsiness state and
        annotate the frame.

        Args:
            bgr_frame : Raw BGR frame from OpenCV VideoCapture.

        Returns:
            annotated_frame : BGR frame with overlaid landmarks & HUD text.
            ear             : Mean EAR of both eyes (0.0 if no face found).
            is_drowsy       : True when the driver is considered drowsy.
        """
        h, w = bgr_frame.shape[:2]

        # Work on a copy so the caller's frame is untouched
        annotated = bgr_frame.copy()

        # -- Convert BGR → RGB for MediaPipe ------------------------------------
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # -- Wrap in mp.Image ---------------------------------------------------
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # -- Run inference ------------------------------------------------------
        result = self._landmarker.detect(mp_image)

        ear = 0.0
        face_found = (
            result.face_landmarks is not None
            and len(result.face_landmarks) > 0
        )

        if face_found:
            landmarks = result.face_landmarks[0]   # First (only) face

            # Extract pixel coordinates for each eye
            left_pixels  = [
                landmark_to_pixel(landmarks[i], w, h) for i in LEFT_EYE_IDX
            ]
            right_pixels = [
                landmark_to_pixel(landmarks[i], w, h) for i in RIGHT_EYE_IDX
            ]

            # Compute per-eye EAR and average them
            left_ear  = compute_ear(left_pixels)
            right_ear = compute_ear(right_pixels)
            ear = (left_ear + right_ear) / 2.0

            # Draw eye outlines (green when awake, red when drowsy)
            eye_color = COLOR_RED if ear < EAR_THRESHOLD else COLOR_GREEN
            draw_eye_landmarks(annotated, left_pixels,  color=eye_color)
            draw_eye_landmarks(annotated, right_pixels, color=eye_color)

            # -- Update drowsiness state machine ----------------------------------
            if ear < EAR_THRESHOLD:
                self._frame_counter += 1
            else:
                # EAR recovered → reset counter and clear alert
                self._frame_counter = 0
                self._is_drowsy = False

            if self._frame_counter >= EAR_CONSEC_FRAMES:
                self._is_drowsy = True
        else:
            # No face detected → treat as unknown, do not reset alert
            self._frame_counter = 0

        # -- Draw HUD -----------------------------------------------------------
        self._draw_hud(annotated, ear, face_found, w, h)

        return annotated, ear, self._is_drowsy

    def release(self) -> None:
        """Release the underlying MediaPipe model resources."""
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _draw_hud(
        self,
        frame: np.ndarray,
        ear: float,
        face_found: bool,
        width: int,
        height: int,
    ) -> None:
        """
        Overlay EAR value, frame counter, face-detection status, and
        (when applicable) a bold drowsiness alert banner.
        """
        # ---- Status line (top-left) ----------------------------------------
        if face_found:
            ear_text    = f"EAR: {ear:.3f}"
            counter_txt = f"Closed frames: {self._frame_counter}/{EAR_CONSEC_FRAMES}"
            ear_color   = COLOR_RED if ear < EAR_THRESHOLD else COLOR_GREEN
            put_text_with_background(
                frame, ear_text, (10, 30),
                font_scale=0.7, text_color=ear_color
            )
            put_text_with_background(
                frame, counter_txt, (10, 65),
                font_scale=0.55, text_color=COLOR_YELLOW
            )
        else:
            put_text_with_background(
                frame, "No face detected", (10, 30),
                font_scale=0.65, text_color=COLOR_YELLOW
            )

        # ---- Threshold reminder (bottom-left) ---------------------------------
        put_text_with_background(
            frame, f"Threshold: {EAR_THRESHOLD}",
            (10, height - 15),
            font_scale=0.5,
            text_color=(180, 180, 180),
            bg_color=(30, 30, 30)
        )

        # ---- Drowsiness alert banner (centre) ---------------------------------
        if self._is_drowsy:
            alert_text = "!! DROWSINESS ALERT !!"

            # Compute text size to centre it
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.1
            thickness  = 3
            (tw, th), _ = cv2.getTextSize(alert_text, font, font_scale, thickness)

            x = (width  - tw) // 2
            y = height // 2

            # Semi-transparent dark background strip
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0,       y - th - 15),
                (width,   y + 15),
                (0, 0, 0),
                cv2.FILLED
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Red alert text
            cv2.putText(
                frame, alert_text, (x, y),
                font, font_scale, COLOR_RED, thickness, cv2.LINE_AA
            )
