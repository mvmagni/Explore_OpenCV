import cv2 as cv
import numpy as np
import os

print (f'OpenCV Version: {cv.__version__}')
print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
print(f'CWD: {os.getcwd()}')

# Resource configs
PROJECT_ROOT_DIR = f'Explore_OpenCV'

# Load in model config and weights
modelConfiguration=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-320.cfg'
#modelConfiguration=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-960.cfg'
modelWeights=f'{PROJECT_ROOT_DIR}/net_configs/yolov3-320.weights'
SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/walk.mp4'
#SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/Produce_0.mp4'

# Coco info
classesFile = f'Explore_OpenCV/net_configs/coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

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

#cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)
#cap.set(cv.CAP_PROP_FRAME_WIDTH,320)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT,180)


# Default target for width and height in YOLO network
whT = 320
hhT = 320
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.3 #lower it is the more aggressive and the less overlapping boxes per object


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIDs = []
    confidence = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores) 
            conf_obj = scores[classID]
            if conf_obj > CONFIDENCE_THRESHOLD:
                # get pixel values
                w = int(detection[2]*wT)
                h = int(detection[3]*hT) 
                
                #x,y is centre point
                x = int((detection[0]*wT) - w/2)
                y = int((detection[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIDs.append(classID)
                confidence.append(float(conf_obj))
                
    #print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox,confidence,CONFIDENCE_THRESHOLD,NMS_THRESHOLD)
    #print(indices)
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv.rectangle(img,(x,y), (x+w,y+h), (255,0,255),2)
        cv.putText(img,f'{classNames[classIDs[i]].upper()} {int(confidence[i]*100)}%',
                (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2
                   )

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
    
    findObjects(outputs,frame)
 
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