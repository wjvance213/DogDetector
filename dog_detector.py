#!/usr/bin/env python3
"""Watch the Iriun Webcam feed and flash "DOG" when a dog is detected, or "Dawg" when a human foot is.

Uses a YOLOv8 detection model (pretrained on COCO, which includes a "dog" class) and a
YOLOv8 pose model whose ankle keypoints are used to spot a human foot in frame.
Press 'q' or Esc to quit.
"""

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"        # nano model: small and fast enough for a laptop CPU
POSE_MODEL_NAME = "yolov8n-pose.pt"  # nano pose model: 17 COCO body keypoints per person
DOG_CLASS_ID = 16                # COCO class index for "dog"
ANKLE_KEYPOINTS = (15, 16)       # COCO keypoint indices for left / right ankle
CONFIDENCE = 0.5                 # minimum detection confidence
KEYPOINT_CONFIDENCE = 0.5        # minimum ankle keypoint confidence to count as a visible foot
DISPLAY_SECONDS = 5.0            # how long "DOG" / "Dawg" stays on screen after a detection
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


def find_feet(pose_model: YOLO, frame) -> list[tuple[int, int]]:
    """Return pixel positions of every confidently visible human ankle in the frame."""
    results = pose_model(frame, conf=CONFIDENCE, verbose=False)
    keypoints = results[0].keypoints
    if keypoints is None or keypoints.conf is None:
        return []

    feet = []
    for xy, conf in zip(keypoints.xy.tolist(), keypoints.conf.tolist()):
        for idx in ANKLE_KEYPOINTS:
            if conf[idx] >= KEYPOINT_CONFIDENCE:
                feet.append((int(xy[idx][0]), int(xy[idx][1])))
    return feet


def draw_labels(frame, labels: list[str]) -> None:
    """Draw each label centred on the frame, stacked vertically."""
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 4, 10  # heavy thickness = bold
    sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text in labels]
    gap = 30
    total_h = sum(th for _, th in sizes) + gap * (len(labels) - 1)
    y = (frame.shape[0] - total_h) // 2
    for text, (tw, th) in zip(labels, sizes):
        y += th
        x = (frame.shape[1] - tw) // 2
        # Black outline behind white fill so it's readable on any background.
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += gap


def main() -> int:
    index = find_camera(CAMERA_NAME)
    if index is None:
        print(f"Camera '{CAMERA_NAME}' not found. Is the Iriun app running and the phone connected?",
              file=sys.stderr)
        return 1

    model = YOLO(MODEL_NAME)            # downloads the weights on first run
    pose_model = YOLO(POSE_MODEL_NAME)  # likewise

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Could not open /dev/video{index} ({CAMERA_NAME}). "
              "Make sure the Iriun app is running and the phone is streaming.", file=sys.stderr)
        return 1

    show_dog_until = 0.0   # timestamp until which "DOG" should be displayed
    show_dawg_until = 0.0  # timestamp until which "Dawg" should be displayed

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera", file=sys.stderr)
            break

        results = model(frame, classes=[DOG_CLASS_ID], conf=CONFIDENCE, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Restart the 5 second window on every detection.
            show_dog_until = time.monotonic() + DISPLAY_SECONDS
            for x1, y1, x2, y2 in boxes.xyxy.int().tolist():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        feet = find_feet(pose_model, frame)
        if feet:
            show_dawg_until = time.monotonic() + DISPLAY_SECONDS
            for x, y in feet:
                cv2.circle(frame, (x, y), 12, (0, 165, 255), 3)

        # Stack the labels so both are readable if a dog and a foot are in frame together.
        labels = []
        if time.monotonic() < show_dog_until:
            labels.append("DOG")
        if time.monotonic() < show_dawg_until:
            labels.append("Dawg")
        draw_labels(frame, labels)

        cv2.imshow("Dog Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
