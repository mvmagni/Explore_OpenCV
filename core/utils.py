import cv2 as cv
import numpy as np
from datetime import datetime
import time
from PIL import Image
from statistics import mean

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
    
    image_show = Image.open(fileName)
    image_show.show()
    
def show_fps(img, prev_frame_time, fps_queue=None):
    queue_length=90
    
    if fps_queue is None:
        fps_queue = []
    
    # font which we will be using to display FPS
    font = cv.FONT_HERSHEY_SIMPLEX
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
 
    # add fps
    fps_queue.append(fps)
    # Remove anything above desired queue/avg length
    while len(fps_queue) > queue_length:
        fps_queue.pop(0)
 
    # putting the FPS count on the frame
    cv.putText(img, str(int(mean((fps_queue)))), (15, 50), font, 1.5, (100, 255, 0), 2, cv.LINE_AA)
    return new_frame_time, fps_queue  

def findObjects(outputs, img, classNames, show_labels=True):
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
        if classIDs[i] == 0: # person
            colour = (255,51,153) # colour=
        elif classIDs[i] == 2: # car
            colour = (255,153,51) # colour=cyan
        else: 
            colour = (0,204,204) # colour=yellow
        cv.rectangle(img,(x,y), (x+w,y+h), colour,2)
        
        if show_labels:
            cv.putText(img,f'{classNames[classIDs[i]].title()} {int(confidence[i]*100)}%',
                    (x,y-5), cv.FONT_HERSHEY_SIMPLEX, 0.6,colour,2)