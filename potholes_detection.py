import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", 'custom', path= '/Users/nanaki/Desktop/best.pt')

# video = cv2.VideoCapture('/Users/nanaki/Desktop/pothole_video/train/rgb/0537.mp4')
video = cv2.VideoCapture('/Users/nanaki/Desktop/pothole_video/test/rgb/0094.mp4')
print(model.names)


while(video.isOpened()):
    succes, frame = video.read()
    
    pred = model(frame)
    
    print(pred)
    print(pred.xyxyn[0])
    
    if len(pred.xyxyn[0]) >= 1: 

        w, h = frame.shape[0], frame.shape[1]
        print(w,h)
        x1 = int(pred.xyxyn[0][0][0].item()*w)
        y1 = int(pred.xyxyn[0][0][1].item()*h)
        x2 = int(pred.xyxyn[0][0][2].item()*w)
        y2 = int(pred.xyxyn[0][0][3].item()*h)
        print(x1, y1, x2, y2)
        
        class_id = int(pred.xyxyn[0][0][0].item())
        class_name = model.names[class_id]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 4)
        cv2.putText(frame, class_name, (x1-10, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)

    cv2.imshow("frame", frame)
    
    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
video.release()

cv2.destroyAllWindows()
    