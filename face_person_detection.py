import sys
import numpy as np
import time
import imutils
import cv2
from matplotlib import pyplot as plt
import torch 

# Mediapipe for face detection
import mediapipe as mp

# Mediapipe for face detection
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap_right = cv2.VideoCapture('/Users/nanaki/Downloads/IMG_1762.MOV')
cap_left = cv2.VideoCapture('/Users/nanaki/Downloads/IMG_1304.MOV')

# Stereo vision setup parameters
baseline = 3.4              #Distance between the cameras [cm]
f = 25              #Camera lense's focal length [mm]
alpha = 120        #Camera field of view in the horisontal plane [degrees]

def calc_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

    return zDepth


face_detection = mp_facedetector.FaceDetection(min_detection_confidence=0.7)

while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    # If cannot catch any frame, break
    if not succes_right or not succes_left:                    
        break
    else:
        start = time.time()
        
        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results_right = face_detection.process(frame_right)
        results_left = face_detection.process(frame_left)

        # Convert the RGB image to BGR
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

        center_right = 0
        center_left = 0

        if results_right.detections:
            for id, detection in enumerate(results_right.detections):
                mp_draw.draw_detection(frame_right, detection)

                bBox = detection.location_data.relative_bounding_box

                w, h, c = frame_right.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        if results_left.detections:
            for id, detection in enumerate(results_left.detections):
                mp_draw.draw_detection(frame_left, detection)

                bBox = detection.location_data.relative_bounding_box

                h, w, c = frame_left.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        if not results_right.detections or not results_left.detections:
            pass

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            depth = calc_depth(center_point_right, center_point_left, frame_right, frame_left, baseline, f, alpha)
            
            # camera error
            depth = depth*0.65
            
            if depth < 0: 
                depth = "Object not detected"
            else:
                depth = str(round(depth,1))
            
            cv2.putText(frame_right, "Distance: " + depth, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_left, "Distance: " + depth, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

        # Show the frames
        cv2.imshow("frame right", frame_right) 
        cv2.imshow("frame left", frame_left)


        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()