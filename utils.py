import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import random
from collections import deque


# function to open the video
def openVideo(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    return cap