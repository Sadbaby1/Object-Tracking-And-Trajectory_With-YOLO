import cv2 as cv
from utils import get_color


def detect_and_track(model, frame):
    """Run YOLO detection and tracking on the current frame."""
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    return results[0]


def center_point(x1, y1, x2, y2):
    """Calculate the center (cx, cy) coordinates of a bounding box."""
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def draw_bounding_box(frame, box, track_id, cls, model, box_thickness=2, font_scale=0.5, font_thickness=2):
    """Draw bounding box, label, and center point on the frame."""
    x1, y1, x2, y2 = map(int, box)
    cx, cy = center_point(x1, y1, x2, y2)
    color = get_color(track_id)

    # Draw bounding box
    cv.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    # Draw label (class name and track ID)
    class_name = model.names[cls]
    label = f"{class_name} ID: {track_id}"
    cv.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        font_thickness
    )

    # Draw circle at the center of the bounding box
    cv.circle(frame, (cx, cy), 3, color, -1)

    return cx, cy, color


