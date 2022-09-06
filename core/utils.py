import cv2 as cv
import numpy as np
from datetime import datetime
import time
from PIL import Image
from statistics import mean
from model_net import ModelNet

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


        show_bounding_box(img=img, 
                          bbox=bbox[i],
                          weight=2,
                          classID=classIDs[i],
                          class_name=f'{classNames[classIDs[i]].title()}',
                          confidence=confidence[i],
                          show_labels=show_labels)

def process_image(modelNet, img, show_labels):
    
    (class_ids, scores, boxes) = modelNet.detect(img)
    
    #print(indices)
    for idx, box in enumerate(boxes, start=0):
        className = modelNet.classes[class_ids[idx]]
        confidence = scores[idx]
        
        
        show_bounding_box(img=img, 
                          bbox=box,
                          classID=class_ids[idx],
                          class_name=f'{className.title()}',
                          confidence=confidence,
                          show_labels=show_labels)    

def show_bounding_box(img, 
                      bbox, 
                      classID, 
                      class_name, 
                      confidence, 
                      show_labels, 
                      weight=2):
    colour=get_class_colour(classID)
    x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
    cv.rectangle(img,(x,y), (x+w,y+h), colour, weight)
    
    if show_labels:
        show_label(img=img,
                   x=x,
                   y=y,
                   class_name=class_name,
                   confidence=confidence,
                   colour=colour)
    
def show_label(img, 
               x,
               y,
               class_name,
               confidence,
               colour):
    cv.putText(img,
               f'{class_name} {int(confidence*100)}%',
               (x,y-5), 
               cv.FONT_HERSHEY_SIMPLEX,
               0.6,
               colour,
               2
               )

def get_class_colour(classID):
    if classID == 0: # person
        colour = (255,51,153) # colour=
    elif classID == 2: # car
        colour = (255,153,51) # colour=cyan
    else: 
        colour = (0,204,204) # colour=yellow
    
    return colour
    