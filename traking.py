import cv2 as cv
from collections import defaultdict, deque
TRAJECTORY_LENGTH = 100
#dictionary for Trajectory history
# create a dictionary to store the trajectory point of each tracked object
trajectory_history = defaultdict(
    
    # Automaticaly create an empty deque when a new track   ID appears
    lambda: deque(
        #store only the last TERAJECTORY_LENGTH positions for each track ID
        maxlen=TRAJECTORY_LENGTH
        )
    )

def draw_trajectory(frame, points, color, line_thickness=3):
    """Draw trajectory trail connecting consecutive center points."""
    pts = list(points)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv.line(frame, pts[i - 1], pts[i], color, line_thickness)
