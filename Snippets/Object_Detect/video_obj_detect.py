import cv2 as cv
import numpy as np
import os


# using mobile SSD
print (f'OpenCV Version: {cv.__version__}')
print(f'CWD: {os.getcwd()}')

# Config video capture. 0 is first
cap = cv.VideoCapture(0)
#cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)


# Default target for width and height in YOLO network
whT = 320

# Coco info
classesFile = f'Snippets/Object_Detect/coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
#print(len(classNames))

# Load in model config and weights
modelConfiguration='Snippets/Object_Detect/yolov3-tiny.cfg'
modelWeights='Snippets/Object_Detect/yolov3-tiny.weights'

# Setup the basics for darknet in CV
net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

img_counter = 0

while True:
    success, frame = cap.read()


    # Convert the frame to a blob for passing to the model
    blob = cv.dnn.blobFromImage(frame, 1/255, (whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    
    
    # Need to get the names of the output layers.
    # This gives the index of the layers, not the names
    # e.g. value of 200 is 199 (because 0 is a valid layer)
    
    
    layerNames = net.getLayerNames()
    #print(layerNames)
    #print(f'layerNames length: {len(layerNames)}, type: {type(layerNames)}')
    #print(net.getUnconnectedOutLayers())
    #print()
    
    #print(f'{layerNames[199]}:{layerNames[226]}:{layerNames[253]}')
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames) (gives output names of the layers)

    outputs = net.forward(outputNames)
    print(len(outputs))

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