import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import random
from collections import deque
from utils import openVideo, get_color

VIDEO_PATH = "02.mp4"
MODEL_PATH = "yolo26n.pt"
BOX_THICKNESS = 2
FONT_SCALE = 0.5
LINE_THICKNESS = 3
FONT_THICKNESS = 2

#load model
model = YOLO(MODEL_PATH)

    

def main():
    cap = openVideo(VIDEO_PATH)
    if cap is None:
        return
    print("Video opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv.resize(frame, (1200, 600))
        
        # Run YOLO model object detection and tracking on current frame.
        results = model.track(frame, # send the current frame to the model for tracking
                              persist=True, #keep the same ID for the same object across frames
                              tracker="bytetrack.yaml", #Use ByteTrack to tgrack the obejects
                              verbose=False # do not print YOLO processing
                              )

        # det first result from the results list
        result = results[0]
        
        #get the bounding boxes, IDs and classes of the detected objects
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            clases = result.boxes.cls.cpu().numpy().astype(int)
            
            #ploting the bounding boxes and IDs on the frame
            for box, track_id, cls in zip(boxes, ids, clases):
                
                # generate center of the bounding box
                x1,y1,x2,y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                #Store trajectory
                #trajectory_history[track_id].append((cx, cy))
                color= get_color(track_id)
                
                # drow bounsing box and ID on the frame
                cv.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                class_name = model.names[cls]
                label = f"{class_name} ID: {track_id}"
                cv.putText(frame,
                           label,
                           (x1, y1 - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 
                           FONT_SCALE, 
                           color, 
                           FONT_THICKNESS)
            
            # drow circle at the center of the bounding box
                cv.circle(frame, (cx, cy), 3, color, -1)
                
            

            
        
        
        #display the image
        cv.imshow("Trajectory tracking", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
    main()
