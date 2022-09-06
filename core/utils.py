import cv2 as cv
import numpy as np
from datetime import datetime
import time
from PIL import Image
from statistics import mean
from model_net import ModelNet

def write_progress_image(img, 
                         operating_config, 
                         extension='jpg', 
                         date_format='%Y-%m-%d_%H-%M-%S'):
    
    fileName = f"{datetime.today().strftime(f'{date_format}_f{operating_config.frame_counter}.{extension}')}"
    writeToFile = f'{operating_config.image_store_dir}/{fileName}'
    
    cv.imwrite(filename=writeToFile, img=img)
    
    image_show = Image.open(writeToFile)
    image_show.show()
    
def show_fps(img, operating_config):
    queue_length=90
    
    if operating_config.fps_queue is None:
        operating_config.fps_queue = []
    
    # font which we will be using to display FPS
    font = cv.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time - operating_config.prev_frame_time)
    operating_config.prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # add fps
    operating_config.fps_queue.append(fps)
    
    # Remove anything above desired queue/avg length
    while len(operating_config.fps_queue) > queue_length:
        operating_config.fps_queue.pop(0)
 
    # putting the FPS count on the frame
    cv.putText(img, 
               str(int(mean((operating_config.fps_queue)))), 
               (15, 50), 
               font, 
               1.5, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)
    

def process_image(modelNet, img, operating_config):
    
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
                          show_labels=operating_config.SHOW_DETECT_LABELS)
        
    
    if operating_config.SHOW_DETECT_LABELS:
        show_fps(img=img, operating_config=operating_config) 

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
    