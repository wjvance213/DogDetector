#!/usr/bin/env python3
"""Watch the Iriun Webcam feed and flash "DOG" on screen when a dog is detected.

Uses a YOLOv8 model (pretrained on COCO, which includes a "dog" class).
Press 'q' or Esc to quit.
"""

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"   # nano model: small and fast enough for a laptop CPU
DOG_CLASS_ID = 16           # COCO class index for "dog"
CONFIDENCE = 0.5            # minimum detection confidence
DISPLAY_SECONDS = 5.0       # how long "DOG" stays on screen after a detection
CAMERA_NAME = "Iriun Webcam"  # v4l2loopback device created by the Iriun client


def find_camera(name: str) -> int | None:
    """Return the index N of /dev/videoN whose V4L2 name matches, or None if absent."""
    for dev in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            if (dev / "name").read_text().strip() == name:
                return int(dev.name.removeprefix("video"))
        except (OSError, ValueError):
            continue
    return None


def main() -> int:
    index = find_camera(CAMERA_NAME)
    if index is None:
        print(f"Camera '{CAMERA_NAME}' not found. Is the Iriun app running and the phone connected?",
              file=sys.stderr)
        return 1

    model = YOLO(MODEL_NAME)  # downloads the weights on first run

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Could not open /dev/video{index} ({CAMERA_NAME}). "
              "Make sure the Iriun app is running and the phone is streaming.", file=sys.stderr)
        return 1

    show_until = 0.0  # timestamp until which "DOG" should be displayed

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera", file=sys.stderr)
            break

        results = model(frame, classes=[DOG_CLASS_ID], conf=CONFIDENCE, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Restart the 5 second window on every detection.
            show_until = time.monotonic() + DISPLAY_SECONDS
            for x1, y1, x2, y2 in boxes.xyxy.int().tolist():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if time.monotonic() < show_until:
            text = "DOG"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale, thickness = 4, 10  # heavy thickness = bold
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = (frame.shape[1] - tw) // 2
            y = (frame.shape[0] + th) // 2
            # Black outline behind white fill so it's readable on any background.
            cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imshow("Dog Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
