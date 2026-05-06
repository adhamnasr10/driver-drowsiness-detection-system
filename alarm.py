# =============================================================================
# alarm.py — Audio alert using the built-in winsound module (Windows only)
# =============================================================================
# The alarm runs in a background daemon thread so it never blocks the
# main video loop.  A simple flag prevents the beep from firing every frame.

import threading
import platform
import sys

from config import ALARM_FREQ_HZ, ALARM_DURATION_MS

# ---------------------------------------------------------------------------
# Platform guard — winsound is Windows-only.
# On other platforms we fall back to a harmless console bell.
# ---------------------------------------------------------------------------
_ON_WINDOWS = platform.system() == "Windows"

if _ON_WINDOWS:
    import winsound


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_alarm_playing: bool = False          # True while a beep thread is running
_alarm_lock = threading.Lock()        # Protects the flag above


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _beep_worker() -> None:
    """
    Target function for the alarm thread.

    Plays a single beep, then clears the _alarm_playing flag so the alarm
    can fire again on the next drowsy event after the driver wakes up.
    """
    global _alarm_playing

    try:
        if _ON_WINDOWS:
            # winsound.Beep(frequency_hz, duration_ms)  — blocking call
            winsound.Beep(ALARM_FREQ_HZ, ALARM_DURATION_MS)
        else:
            # Fallback: write the ASCII BEL character to the terminal
            sys.stdout.write("\a")
            sys.stdout.flush()
    finally:
        # Always release the flag, even if an exception occurred
        with _alarm_lock:
            _alarm_playing = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def trigger_alarm() -> None:
    """
    Trigger the drowsiness alarm in a background thread.

    Safe to call every frame — it is a no-op while a beep is already
    playing, preventing overlapping / rapid-fire beeps.
    """
    global _alarm_playing

    with _alarm_lock:
        if _alarm_playing:
            return                    # Already beeping; skip this call
        _alarm_playing = True

    # Daemon thread: automatically killed when the main program exits
    t = threading.Thread(target=_beep_worker, daemon=True)
    t.start()


def reset_alarm() -> None:
    """
    Manually reset the alarm flag (e.g. when the driver is detected as awake).

    Normally the flag resets itself after each beep completes, but this
    function can be called explicitly if needed.
    """
    global _alarm_playing
    with _alarm_lock:
        _alarm_playing = False


def is_alarm_active() -> bool:
    """Return True if the alarm beep thread is currently running."""
    with _alarm_lock:
        return _alarm_playing
