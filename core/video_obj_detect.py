import cv2 as cv
import numpy as np
import os
import utils

print (f'OpenCV Version: {cv.__version__}')
print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
print(f'CWD: {os.getcwd()}')

# Resource configs
SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/walk.mp4'
#SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/Produce_0.mp4'

PROJECT_ROOT_DIR = f'{os.getcwd()}/Explore_OpenCV'

# Load in model config and weights
modelConfiguration=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-320.cfg'
#modelConfiguration=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-960.cfg'
modelWeights=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-320.weights'

# Coco info
classNames = utils.get_classNames(f'{PROJECT_ROOT_DIR}/net_configs/coco.names')
if classNames is None:
    print(f'classNames not loaded')
    exit()

#######################################################################
# Setup the NN parameters
# Setup the basics for darknet in CV
net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
#net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Need to get the names of the output layers.
# This gives the index of the layers, not the names
# e.g. value of 200 is 199 (because 0 is a valid layer)
layerNames = net.getLayerNames()
print(layerNames)
print(f'layerNames length: {len(layerNames)}, type: {type(layerNames)}')
print(net.getUnconnectedOutLayers())

outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
print(outputNames) #(gives output names of the layers)
########################################################################

# Config video capture. 0 is first
# Added cv.CAP_DSHOW to avoid several minute lag of opening cam on windows
# No lag opening on Linux
cap = cv.VideoCapture(0,cv.CAP_DSHOW)
#cap = cv.VideoCapture(SAMPLE_VIDEO)

cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)

# Default target for width and height in YOLO network
whT = 320
hhT = 320

img_counter = 0
frame_counter = 0
while True:
    frame_counter += 1
    success, frame = cap.read()
    #print(frame.shape)


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
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1


# Release the cam link
cap.release()

# Clear all the windows
cv.destroyAllWindows()