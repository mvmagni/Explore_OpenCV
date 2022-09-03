import cv2 as cv
import numpy as np
import os
import utils
import yolo_config as yc

print(f'OpenCV Version: {cv.__version__}')
print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
print(f'CWD: {os.getcwd()}')

# Resource configs
PROJECT_ROOT_DIR = f'{os.getcwd()}/Explore_OpenCV'
IMAGE_STORE_DIR = f'd:/OBS_Recordings'
SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/walk.mp4'

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
modelConfiguration=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-320.cfg'
modelWeights=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-320.weights'
net, outputNames = yc.get_net_config(modelConfiguration, modelWeights)
###############################################################################

###############################################################################
# Config video capture. 0 is first
# Added cv.CAP_DSHOW to avoid several minute lag of opening cam on windows
# No lag opening on Linux
cap = cv.VideoCapture(0,cv.CAP_DSHOW)
#cap = cv.VideoCapture(SAMPLE_VIDEO)

cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)
################################################################################


# Default target for width and height in YOLO network
# Used in cv.dnn.blobFromImage
whT = 320
hhT = 320

frame_counter = 0
while True:
    frame_counter += 1
    success, frame = cap.read()

    if frame_counter == 1:
        print(f'Image size: {frame.shape}')

    # Only process every second frame through the net
    #if frame_counter % 1 == 0:
    # Convert the frame to a blob for passing to the model
    blob = cv.dnn.blobFromImage(frame, 1/255, (whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    outputs = net.forward(outputNames)
    # print(outputs[0].shape) #300, 85 : 300 is bounding boxes
    # print(outputs[1].shape) #1200, 85 : 1200 is bounding boxes
    # print(outputs[2].shape) #4800, 85 : 4800 is bounding boxes
    # print(outputs[0][0])
    #Breakdown of 85
    # 1-4 are center x, centery, width, height
    # 5 is confidence there is something in the bounding box
    # other 80 represent prob of the 80 original coco classes
    
    utils.findObjects(outputs,frame,classNames)
 
    # Show the image
    cv.imshow('OpenCV Test',frame)

    # Esc to close, space to write a copy of the image
    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        utils.write_progress_image(img=frame,
                                   directory=IMAGE_STORE_DIR,
                                   frame_count=frame_counter)        


# Release the cam link
cap.release()

# Clear all the windows
cv.destroyAllWindows()