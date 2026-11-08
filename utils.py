import cv2 as cv
import random

# function to open the video
def openVideo(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap


track_colors = {}

# function to generate a random color for each track ID
def get_color(track_id):
    # check whether this track ID already has a color assigned to it
    if track_id not in track_colors:
        # generate a random color for this track ID
        track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return track_colors[track_id]
