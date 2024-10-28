import cv2
import time
import torch
import numpy as np
from models.experimental import attempt_load  # Ensure models.experimental is accessible
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load the YOLOv7 Tiny model
weights_path = '/home/pi/Desktop/Sensing/yolov7/yolov7-tiny.pt'
device = select_device('cpu')  # Use CPU on Raspberry Pi
model = attempt_load(weights_path, map_location=device)  # Load model with appropriate device

# Set the video source to your USB camera (usually /dev/video0)
video_source = 0  # Change this if you have multiple cameras
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Start time for 10 seconds limit
start_time = time.time()
duration = 10  # Duration in seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Prepare frame for YOLOv7
    img = cv2.resize(frame, (640, 640))  # Resize frame to 640x640
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        pred = model(img, augment=False)
        
        # Check if pred is a tuple and extract the tensor if so
        if isinstance(pred, tuple):
            pred = pred[0]  # Assuming the first element is the tensor

        # Apply non-max suppression on predictions
        detections = non_max_suppression(pred)

    # Process detections
    for i, det in enumerate(detections):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{int(cls)} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv7 Real-Time Detection', frame)

    # Check if 10 seconds have passed
    if time.time() - start_time > duration:
        print("Detection completed for 10 seconds.")
        break

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
