import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Choose desired YOLOv5 model

# Open both cameras
cap_right = cv2.VideoCapture('/Users/nanaki/Downloads/IMG_1749.MOV')
cap_left = cv2.VideoCapture('/Users/nanaki/Downloads/IMG_1295.MOV')

# Stereo vision setup parameters
baseline = 3.4  # Distance between the cameras [cm]
f = 25          # Camera lens's focal length [mm]
alpha = 130     # Camera field of view in the horizontal plane [degrees]

# Function to calculate depth using stereo vision
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
    disparity = x_left - x_right  # Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline * f_pixel) / disparity  # Depth in [cm]
    return zDepth


# Main program loop
while cap_right.isOpened() and cap_left.isOpened():
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    # If cannot capture any frame, break
    if not success_right or not success_left:
        break

    # Convert the BGR image to RGB
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

    # Perform object detection on the right frame
    results_right = model(frame_right)

    # Get detected objects' bounding boxes and confidence scores
    boxes_right = results_right.xyxy[0][:, :4].cpu().numpy()  # (xmin, ymin, xmax, ymax)
    confidences_right = results_right.xyxy[0][:, 4].cpu().numpy()

    # Perform object detection on the left frame
    results_left = model(frame_left)

    # Get detected objects' bounding boxes and confidence scores
    boxes_left = results_left.xyxy[0][:, :4].cpu().numpy()  # (xmin, ymin, xmax, ymax)
    confidences_left = results_left.xyxy[0][:, 4].cpu().numpy()

    # Loop through each detected object in the right frame
    for box_right, confidence_right in zip(boxes_right, confidences_right):
        # Loop through each detected object in the left frame
        for box_left, confidence_left in zip(boxes_left, confidences_left):
            # Check if objects overlap
            if (box_right[0] < box_left[2] and box_right[2] > box_left[0] and
                    box_right[1] < box_left[3] and box_right[3] > box_left[1]):
                
                h, w, c = frame_right.shape
                
                box_right[0], box_right[1], box_right[2], box_right[3] = int(box_right[0]*w), int(box_right[1]*h), int(box_right[2]*w), int(box_right[3]*h)
                
                # Calculate center points of the detected objects
                center_point_right = ((box_right[0] + box_right[2]) / 2, (box_right[1] + box_right[3]) / 2)
                center_point_left = ((box_left[0] + box_left[2]) / 2, (box_left[1] + box_left[3]) / 2)

                # Calculate depth for the object
                depth = calc_depth(center_point_right, center_point_left, frame_right, frame_left, baseline, f, alpha)

                # Display depth on the right frame
                cv2.putText(frame_right, f'Depth: {depth:.2f} cm', (int(box_right[0]), int(box_right[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display depth on the left frame
                cv2.putText(frame_left, f'Depth: {depth:.2f} cm', (int(box_left[0]), int(box_left[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frames
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Left Camera", frame_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
