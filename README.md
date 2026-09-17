# Dog Detector

Watches the Iriun Webcam feed (phone camera) and shows a bold **DOG** for 5 seconds whenever a dog is in frame.

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

The YOLOv8 nano weights (~6 MB) download automatically on first run.
Press `q` or Esc to quit.
