# Driver Drowsiness Detection System

Real-time drowsiness detection using Python, OpenCV, and the **MediaPipe Tasks API** (Face Landmarker).  
No dlib. No external audio libraries. Pure Python standard-library alarm via `winsound`.

---

## Project Structure

```
drowsiness_project/
├── main.py          ← Entry point / video loop
├── detector.py      ← MediaPipe Face Landmarker + EAR state machine
├── utils.py         ← EAR computation & drawing helpers
├── alarm.py         ← Threaded winsound alarm
├── config.py        ← All tunable parameters
├── requirements.txt ← Python dependencies
└── face_landmarker.task  ← (download — see Step 3 below)
```

---

## How It Works

1. **Capture** — OpenCV grabs each webcam frame.  
2. **Detect** — MediaPipe Face Landmarker locates 478 facial landmarks.  
3. **EAR** — Eye Aspect Ratio is computed from 6 landmarks per eye:

   ```
   EAR = (||p1−p5|| + ||p2−p4||) / (2 · ||p0−p3||)
   ```

   A fully open eye → EAR ≈ 0.30+  
   A closed eye → EAR ≈ 0.0  

4. **Alert** — When EAR stays below `EAR_THRESHOLD` (default 0.25) for  
   `EAR_CONSEC_FRAMES` (default 20) consecutive frames, a **DROWSINESS ALERT**  
   is displayed and a beep sounds.

---

## Setup Instructions

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 – 3.12 | MediaPipe 0.10.x wheel availability; see note below |
| pip | Bundled with Python |
| Webcam | USB or built-in |
| Windows OS | Required for `winsound`; see cross-platform note |

> **Python 3.14 note**  
> MediaPipe does not yet publish binary wheels for Python 3.14 on PyPI.  
> Use Python **3.10, 3.11, or 3.12** until upstream wheels are available.  
> You can manage multiple Python versions with [pyenv-win](https://github.com/pyenv-win/pyenv-win).

---

### Step 1 — Clone / create the project folder

```bash
# If you received a zip, unzip it; or create the folder yourself
cd drowsiness_project
```

---

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Download the MediaPipe Face Landmarker model

The model file (`face_landmarker.task`) must be downloaded separately and placed
**in the same folder as `main.py`**.

**Option A — curl (Windows / macOS / Linux)**

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**Option B — wget (Linux / macOS)**

```bash
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**Option C — Python one-liner**

```python
python -c "
import urllib.request
url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
urllib.request.urlretrieve(url, 'face_landmarker.task')
print('Downloaded face_landmarker.task')
"
```

Verify the file exists:

```bash
# Windows
dir face_landmarker.task

# macOS / Linux
ls -lh face_landmarker.task
```

---

### Step 5 — Run the system

```bash
python main.py
```

A window titled **"Driver Drowsiness Detection"** will open showing the webcam feed.

| Key | Action |
|---|---|
| `ESC` | Quit the application |
| Close × | Also quits cleanly |

---

## Configuration

Edit `config.py` to tune behaviour:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | EAR below this → eye considered closed |
| `EAR_CONSEC_FRAMES` | `20` | Frames before alert fires (~0.7 s at 30 fps) |
| `CAMERA_INDEX` | `0` | Change to `1`, `2` … for a different camera |
| `ALARM_FREQ_HZ` | `1000` | Beep frequency |
| `ALARM_DURATION_MS` | `500` | Beep length |

---

## Cross-Platform Notes

| Platform | Sound | Notes |
|---|---|---|
| **Windows** | `winsound.Beep` | Full support |
| **macOS / Linux** | ASCII `\a` bell | Terminal bell only; install `beepy` or `pygame` for a real tone |

On macOS/Linux the alarm fallback in `alarm.py` writes the BEL character — you may need to enable terminal sound.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open camera` | Check another app isn't using the webcam; try `CAMERA_INDEX = 1` |
| `Failed to load model` | Confirm `face_landmarker.task` is in the project folder |
| `No module named mediapipe` | Run `pip install -r requirements.txt` in the active venv |
| EAR always 0 | Face not detected — improve lighting, move closer to camera |
| Alert never fires | Lower `EAR_THRESHOLD` to `0.20` or reduce `EAR_CONSEC_FRAMES` |
| Alert fires too often | Raise `EAR_THRESHOLD` to `0.28` or increase `EAR_CONSEC_FRAMES` |
