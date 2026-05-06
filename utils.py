# =============================================================================
# utils.py — Utility functions: EAR computation & landmark visualisation
# =============================================================================

import math
import cv2
import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Point2D = Tuple[float, float]   # (x, y) in pixel or normalised coordinates


# ---------------------------------------------------------------------------
# Eye Aspect Ratio
# ---------------------------------------------------------------------------
def euclidean_distance(p1: Point2D, p2: Point2D) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_ear(eye_points: List[Point2D]) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) from 6 landmark points.

    The 6 points follow the standard layout:
        p0 = left corner  (index 0)
        p1 = upper-left   (index 1)
        p2 = upper-right  (index 2)
        p3 = right corner (index 3)
        p4 = lower-right  (index 4)
        p5 = lower-left   (index 5)

    Formula (Soukupová & Čech, 2016):
        EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)

    A fully open eye gives EAR ≈ 0.3+.
    A closed eye gives EAR close to 0.

    Args:
        eye_points: List of exactly 6 (x, y) tuples.

    Returns:
        EAR as a float.  Returns 0.0 if input is invalid.
    """
    if len(eye_points) != 6:
        return 0.0

    p0, p1, p2, p3, p4, p5 = eye_points

    # Vertical distances
    vert_a = euclidean_distance(p1, p5)   # upper-left  ↔ lower-left
    vert_b = euclidean_distance(p2, p4)   # upper-right ↔ lower-right

    # Horizontal distance
    horiz  = euclidean_distance(p0, p3)   # left corner ↔ right corner

    if horiz == 0.0:
        return 0.0

    ear = (vert_a + vert_b) / (2.0 * horiz)
    return ear


# ---------------------------------------------------------------------------
# Landmark visualisation helpers
# ---------------------------------------------------------------------------
def landmark_to_pixel(
    landmark,
    frame_width: int,
    frame_height: int
) -> Point2D:
    """
    Convert a MediaPipe normalised landmark (0-1 range) to pixel coordinates.

    Args:
        landmark : A mediapipe NormalizedLandmark object (has .x and .y).
        frame_width  : Width  of the video frame in pixels.
        frame_height : Height of the video frame in pixels.

    Returns:
        (px, py) tuple of integer pixel coordinates.
    """
    px = int(landmark.x * frame_width)
    py = int(landmark.y * frame_height)
    return (px, py)


def draw_eye_landmarks(
    frame: np.ndarray,
    eye_pixels: List[Point2D],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
    draw_outline: bool = True
) -> None:
    """
    Draw circles at each eye landmark and optionally connect them with lines.

    Args:
        frame        : The BGR video frame to draw on (modified in-place).
        eye_pixels   : List of (x, y) pixel positions for the 6 eye landmarks.
        color        : BGR colour for circles and lines.
        radius       : Radius of each landmark dot.
        draw_outline : If True, connect the dots with line segments.
    """
    if not eye_pixels:
        return

    # Draw individual dots
    for (px, py) in eye_pixels:
        cv2.circle(frame, (px, py), radius, color, -1)

    # Connect dots to form the eye outline
    if draw_outline and len(eye_pixels) == 6:
        pts = np.array(eye_pixels, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def put_text_with_background(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> None:
    """
    Render text with a filled rectangle background for readability.

    Args:
        frame      : BGR frame to draw on (in-place).
        text       : String to display.
        origin     : Bottom-left corner of the text (x, y).
        font_scale : OpenCV font scale factor.
        thickness  : Text stroke thickness.
        text_color : BGR text colour.
        bg_color   : BGR background rectangle colour.
        padding    : Pixels of padding around the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - h - padding),
        (x + w + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED
    )
    # Draw text on top
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness,
                cv2.LINE_AA)
