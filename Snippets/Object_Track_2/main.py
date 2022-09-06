import cv2 as cv
import os
from tracker import *


print(f'OpenCV Version: {cv.__version__}')
print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
print(f'CWD: {os.getcwd()}')

# Resource configs
PROJECT_ROOT_DIR = f'{os.getcwd()}/Explore_OpenCV'
SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/highway.mp4'
#SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/walk.mp4'


# Create tracker object
tracker = EuclideanDistTracker()
cap = cv.VideoCapture(SAMPLE_VIDEO)

# Object detection from Stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[340: 720,500: 800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > 100:
            #cv.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)


            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("roi", roi)
    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(30)
    if key == 27: # ESC key
        break

cap.release()
cv.destroyAllWindows()