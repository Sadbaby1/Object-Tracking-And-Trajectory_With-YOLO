import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import random
from collections import deque
from utils import openVideo

video_path = "01.mp4"

def main():
    cap = openVideo(video_path)
    print("Video opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv.resize(frame, (640, 480))
    
        
        #display the image
        cv.imshow("Video", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
