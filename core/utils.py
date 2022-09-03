import cv2 as cv
import numpy as np
from datetime import datetime


CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.3 #lower it is the more aggressive and the less overlapping boxes per object

def get_classNames(classFile):
    # Coco info
    classNames = []

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    return classNames

def write_progress_image(img, directory, frame_count, extension='jpg', date_format='%Y-%m-%d_%H-%M-%S'):
    fileName = f"{directory}/{datetime.today().strftime(f'{date_format}_f{frame_count}.{extension}')}"
    cv.imwrite(filename=fileName, img=img)

def show_fps():
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
 
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
 
    

def findObjects(outputs, img, classNames):
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