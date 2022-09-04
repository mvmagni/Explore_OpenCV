import cv2 as cv
import numpy as np
import os
import utils
import yolo_config as yc
from statistics import mean
import time

print(f'OpenCV Version: {cv.__version__}')
print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
print(f'CWD: {os.getcwd()}')

# Resource configs
PROJECT_ROOT_DIR = f'{os.getcwd()}/Explore_OpenCV'
IMAGE_STORE_DIR = f'd:/OBS_Recordings'
SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/walk.mp4'

#Configs for changing video while running
SHOW_FPS=True
SHOW_DETECT=True
SHOW_DETECT_LABELS=True

###############################################################################
# Get coco info
classNames = utils.get_classNames(f'{PROJECT_ROOT_DIR}/net_configs/coco.names')
if classNames is None:
    print(f'classNames not loaded')
    exit()
###############################################################################

###############################################################################
# Setup the NN parameters
# Setup the basics for darknet in CV
# Load in model config and weights
#CONFIG_TYPE=yc.MODEL_YOLOV3_320_320

CONFIG_TYPE=yc.MODEL_YOLOV3_320_192
#CONFIG_TYPE=yc.MODEL_YOLOV3_416_256
#CONFIG_TYPE=yc.MODEL_YOLOV3_576_352
#CONFIG_TYPE=yc.MODEL_YOLOV3_608_352

#CONFIG_TYPE=yc.MODEL_YOLOV3T_320_192
#CONFIG_TYPE=yc.MODEL_YOLOV3T_416_256
#CONFIG_TYPE=yc.MODEL_YOLOV3T_576_352
#CONFIG_TYPE=yc.MODEL_YOLOV3T_608_352

#CONFIG_TYPE=yc.MODEL_YOLOV4N_320_192
#CONFIG_TYPE=yc.MODEL_YOLOV4N_416_256
#CONFIG_TYPE=yc.MODEL_YOLOV4N_576_352
#CONFIG_TYPE=yc.MODEL_YOLOV4N_608_352

#CONFIG_TYPE=yc.MODEL_YOLOV4T_320_192
#CONFIG_TYPE=yc.MODEL_YOLOV4T_416_256
#CONFIG_TYPE=yc.MODEL_YOLOV4T_576_352
#CONFIG_TYPE=yc.MODEL_YOLOV4T_608_352
    
net, outputNames, whT, hhT = yc.get_net_config(model_type=CONFIG_TYPE,
                                               config_dir=f'{PROJECT_ROOT_DIR}/net_configs')
###############################################################################

###############################################################################
# Config video capture. 0 is first
# Added cv.CAP_DSHOW to avoid several minute lag of opening cam on windows
# No lag opening on Linux
#cap = cv.VideoCapture(0,cv.CAP_DSHOW)
cap = cv.VideoCapture(SAMPLE_VIDEO)
fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)
################################################################################

fps_queue=None
prev_frame_time = 0
frame_counter = 0
while True:
    frame_counter += 1
    success, frame = cap.read()

    if frame_counter == 1:
        print(f'Image size: {frame.shape}')

    if SHOW_DETECT:
        blob = cv.dnn.blobFromImage(frame, 1/255, (whT,hhT),[0,0,0],1,crop=False)
         
        net.setInput(blob)
        outputs = net.forward(outputNames)
            
        utils.findObjects(outputs,frame,classNames, show_labels=SHOW_DETECT_LABELS)

    if SHOW_FPS:
        prev_frame_time, fps_queue = utils.show_fps(frame, prev_frame_time, fps_queue)

    # Show the image
    cv.imshow('OpenCV Test',frame)

    # Esc to close, space to write a copy of the image
    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 102: #small f
        SHOW_FPS = not SHOW_FPS
    elif k%256 == 100: #small d
        SHOW_DETECT = not SHOW_DETECT
    elif k%256 == 108: # small l (letter L)
        SHOW_DETECT_LABELS = not SHOW_DETECT_LABELS
    elif k%256 == 32:
        # SPACE pressed
        utils.write_progress_image(img=frame,
                                   directory=IMAGE_STORE_DIR,
                                   frame_count=frame_counter)        


# Release the cam link
cap.release()

# Clear all the windows
cv.destroyAllWindows()