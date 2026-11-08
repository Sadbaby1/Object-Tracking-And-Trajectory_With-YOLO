import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import random
from collections import deque
from utils import openVideo

video_path = "02.mp4"

def main():
    cap = openVideo(video_path)
    if cap is None:
        return
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
    
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
    main()
