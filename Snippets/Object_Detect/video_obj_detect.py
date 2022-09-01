import cv2 as cv
import numpy as np
import os


print (f'OpenCV Version: {cv.__version__}')
print(f'CWD: {os.getcwd()}')

# Resource configs
PROJECT_ROOT_DIR = f'Explore_OpenCV'

# Load in model config and weights
modelConfiguration=f'{PROJECT_ROOT_DIR}/Snippets/Object_Detect/yolov3-320.cfg'
modelWeights=f'{PROJECT_ROOT_DIR}/Snippets/Object_Detect/yolov3-320.weights'

# Coco info
classesFile = f'Explore_OpenCV/Snippets/Object_Detect/coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#######################################################################
# Setup the NN parameters
# Setup the basics for darknet in CV
net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

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

#cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)
cap.set(cv.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,180)


# Default target for width and height in YOLO network
whT = 320

img_counter = 0

while True:
    success, frame = cap.read()
    print(frame.shape)

    # Convert the frame to a blob for passing to the model
    blob = cv.dnn.blobFromImage(frame, 1/255, (whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    outputs = net.forward(outputNames)

    # Show the image
    cv.imshow('Image',frame)

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