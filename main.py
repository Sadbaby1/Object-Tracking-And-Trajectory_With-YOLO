import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from collections import defaultdict, deque
from utils import openVideo
from detection import detect_and_track, draw_bounding_box

VIDEO_PATH = "02.mp4"
MODEL_PATH = "yolo26n.pt"
BOX_THICKNESS = 2
FONT_SCALE = 0.5
LINE_THICKNESS = 3
FONT_THICKNESS = 2

# Load YOLO model
model = YOLO(MODEL_PATH)


def main():
    cap = openVideo(VIDEO_PATH)
    if cap is None:
        return
    print("Video opened successfully.")

    # Dictionary storing trajectory history: track_id -> deque of center points
    trajectory_history = defaultdict(lambda: deque(maxlen=30))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv.resize(frame, (1200, 600))

        # Run YOLO model object detection and tracking on current frame
        result = detect_and_track(model, frame)

        # Get the bounding boxes, IDs and classes of the detected objects
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            clases = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, clases):
                # Draw bounding box, label and center point
                cx, cy, color = draw_bounding_box(
                    frame, box, track_id, cls, model,
                    box_thickness=BOX_THICKNESS,
                    font_scale=FONT_SCALE,
                    font_thickness=FONT_THICKNESS
                )

        # Update trajectory history for each track ID        
        # Display the frame
        cv.imshow("Trajectory tracking", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
