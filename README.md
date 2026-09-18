# Dog Detector

Watches the Iriun Webcam feed (phone camera) and shows a bold **DOG** for 5 seconds whenever a dog is in frame,
and **Dawg** for 5 seconds whenever a human foot is in frame (detected via the YOLOv8 pose model's ankle keypoints).

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Start the Iriun Webcam app on your laptop and connect your phone first — the script looks for the
V4L2 device named `Iriun Webcam` and exits if it isn't present.

```
.venv/bin/python dog_detector.py
```

The YOLOv8 nano detection and pose weights (~6.5 MB each) download automatically on first run.
Press `q` or Esc to quit.
