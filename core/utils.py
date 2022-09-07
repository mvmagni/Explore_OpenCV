import cv2 as cv
import numpy as np
from datetime import datetime
import time
from PIL import Image
from statistics import mean
from model_net import ModelNet
import yolo_config as yc

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
               'FPS:'+ str(int(mean((operating_config.fps_queue)))), 
               (15, 50), 
               font, 
               1.0, 
               (100, 255, 0), 
               1, 
               cv.LINE_AA)
    
def write_info_bottom_left(img,
                           info,
                           operating_config):
    h, w, c = img.shape
    # print(f'w:{w}, h:{h}')
    cv.putText(img, 
               info, 
               (40, h-50), 
               operating_config.font, 
               1.5, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)

def process_image(img, operating_config):
   
    if operating_config.SHOW_FPS:
        show_fps(img=img, operating_config=operating_config)
    
    if operating_config.SHOW_MODEL_CONFIG:
        write_info_bottom_left(img=img,
                               info=operating_config.modelNet.model_type,
                               operating_config=operating_config)

    if operating_config.SHOW_DETECT:
        (class_ids, scores, boxes) = operating_config.modelNet.detect(img)
        
        #print(indices)
        for idx, box in enumerate(boxes, start=0):
            className = operating_config.modelNet.classes[class_ids[idx]]
            confidence = scores[idx]
            
            show_bounding_box(img=img, 
                            bbox=box,
                            classID=class_ids[idx],
                            class_name=f'{className.title()}',
                            confidence=confidence,
                            show_labels=operating_config.SHOW_DETECT_LABELS)
            
            
def show_bounding_box(img, 
                      bbox, 
                      classID, 
                      class_name, 
                      confidence, 
                      show_labels, 
                      weight=1):
    colour=get_class_colour(classID)
    x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
    
    cv.rectangle(img,(x,y), (x+w,y+h), colour, weight)
    
    line_width_max = 50
    line_width = min(int(w/2 * 0.30), line_width_max)
    line_thickness_w = 3
    line_thickness_h = 3
    
    # Top left
    cv.line(img, (x,y), (x + line_width, y), colour, thickness=line_thickness_w)
    cv.line(img, (x,y), (x, y + line_width), colour, thickness=line_thickness_h)
    
    # Top right
    cv.line(img, (x + w,y), (x + w - line_width, y), colour, thickness=line_thickness_w)
    cv.line(img, (x + w,y), (x + w, y + line_width), colour, thickness=line_thickness_h)
    
    # Bottom left
    cv.line(img, (x,y + h), (x + line_width, y + h), colour, thickness=line_thickness_w)
    cv.line(img, (x,y + h), (x, y + h - line_width), colour, thickness=line_thickness_h)
    
    # Bottom right
    cv.line(img, (x + w, y + h), (x + w - line_width, y + h), colour, thickness=line_thickness_w)
    cv.line(img, (x + w, y + h), (x + w, y + h - line_width), colour, thickness=line_thickness_h)
    
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
        colour = (255,51,153) # colour=blue
    elif classID == 2: # car
        colour = (255,153,51) # colour=cyan
    else: 
        colour = (0,204,204) # colour=yellow
    
    return colour


def handle_config_key_input(img,
                            key,
                            operating_config):
    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        operating_config.RUN_PROGRAM = False
        operating_config.PROCESS_IMAGES = False
        operating_config.CONFIGURE = False
        
    elif key%256 == 103: # letter g, Go and process video
        print(f'Running process')
        operating_config.CONFIGURE = False
        operating_config.PROCESS_IMAGES = True
    
    # Model changes - cycle through available
    elif key%256 == 91: # left square bracket "["
        # Set previous model as desired one
        set_desired_model(operating_config, -1)  
    elif key%256 == 93: # right square bracket "]"
        # Set next model as desired one
        set_desired_model(operating_config, 1)
    
    # Config changes - can be done in runtime
    elif key%256 == 102: #small f
        operating_config.SHOW_FPS = not operating_config.SHOW_FPS
    elif key%256 == 100: #small d
        operating_config.SHOW_DETECT = not operating_config.SHOW_DETECT
    elif key%256 == 108: # small l (letter L)
        operating_config.SHOW_DETECT_LABELS = not operating_config.SHOW_DETECT_LABELS
    
    # Increase or decrease confidence threshold
    elif key%256 == 43: # + increase confidence    
        adjust_confidence_threshold(operating_config=operating_config,
                                    adjust_by=operating_config.CONF_THRESHOLD_ADJUSTBY)
    elif key%256 == 45: # - decrease confidence
        adjust_confidence_threshold(operating_config=operating_config,
                                    adjust_by=-operating_config.CONF_THRESHOLD_ADJUSTBY)
    
    # Increase or decrease confidence threshold
    elif key%256 == 39: # ' increase nms_threshold    
        adjust_nms_threshold(operating_config=operating_config,
                             adjust_by=operating_config.NMS_THRESHOLD_ADJUSTBY)
        
    elif key%256 == 59: # ; decrease nms_threshold
        adjust_nms_threshold(operating_config=operating_config,
                             adjust_by=-operating_config.NMS_THRESHOLD_ADJUSTBY)
    
    # Take screenshot
    elif key%256 == 32: # SPACE pressed
        write_progress_image(img=img,
                             operating_config=operating_config
                            )   

def handle_processing_key_input(img,
                                key,
                                operating_config):
    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        operating_config.RUN_PROGRAM = False
        operating_config.PROCESS_IMAGES = False
        operating_config.CONFIGURE = False

    elif key%256 == 99: #small c
        operating_config.PROCESS_IMAGES = False
        operating_config.CONFIGURE = True
    elif key%256 == 102: #small f
        operating_config.SHOW_FPS = not operating_config.SHOW_FPS
    elif key%256 == 100: #small d
        operating_config.SHOW_DETECT = not operating_config.SHOW_DETECT
    elif key%256 == 108: # small l (letter L)
        operating_config.SHOW_DETECT_LABELS = not operating_config.SHOW_DETECT_LABELS
    elif key%256 == 32:
        # SPACE pressed
        write_progress_image(img=img,
                             operating_config=operating_config
                            )

def adjust_nms_threshold(operating_config,
                         adjust_by):
    
    new_threshold = round(operating_config.NMS_THRESHOLD + adjust_by,2)
    
    if new_threshold < 0: #Can't go below 0
        # Adding the operating config threshold stays if model changes
        operating_config.NMS_THRESHOLD = 0
        operating_config.modelNet.nms_threshold = 0
    elif new_threshold > 1: #Can't go above 1
        # Adding the operating config threshold stays if model changes
        operating_config.NMS_THRESHOLD = 1
        operating_config.modelNet.nms_threshold = 1
    else:
        # Adding the operating config threshold stays if model changes
        operating_config.NMS_THRESHOLD = new_threshold
        operating_config.modelNet.nms_threshold = new_threshold


def adjust_confidence_threshold(operating_config,
                                adjust_by):
    
    new_threshold = round(operating_config.CONFIDENCE_THRESHOLD + adjust_by,2)
    
    if new_threshold < 0: #Can't go below 0
        # Adding the operating config threshold stays if model changes
        operating_config.CONFIDENCE_THRESHOLD = 0
        operating_config.modelNet.confidence_threshold = 0
    elif new_threshold > 1: #Can't go above 1
        # Adding the operating config threshold stays if model changes
        operating_config.CONFIDENCE_THRESHOLD = 1
        operating_config.modelNet.confidence_threshold = 1
    else:
        # Adding the operating config threshold stays if model changes
        operating_config.CONFIDENCE_THRESHOLD = new_threshold
        operating_config.modelNet.confidence_threshold = new_threshold


def set_desired_model(operating_config,
                      increment):
    desired_model=operating_config.detection_model
    model_list = yc.MODEL_LIST
    
    curr_model_index=model_list.index(desired_model)    
    
    # deal with edge cases first
    if curr_model_index == (len(model_list)-1) and increment == 1: # Case end of list
        print(f'Current index at end of list')
        desired_model=model_list[0]
    elif curr_model_index == 0 and increment == -1: # Case start of list
        print(f'Current index at start of list')
        desired_model=model_list[len(model_list)-1]
    else:
        print(f'Standard case: increment by: {increment}')
        desired_model=model_list[curr_model_index+increment]
    
    operating_config.detection_model=desired_model
    print(f'Desired model: {desired_model}')

def write_row_of_info(img,
                      info,
                      rownum,
                      operating_config):
    cv.putText(img, 
               info, 
               (40, 60 * rownum), 
               operating_config.font, 
               1.0, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)

def write_config_screen(img,
                        operating_config):
        
        screen_info=[]
        screen_info.append(f'Detection model active:  {operating_config.modelNet.model_type}')
        screen_info.append(f'Detection model desired: {operating_config.detection_model}')
        screen_info.append(f'Detection on:     {operating_config.SHOW_DETECT}')
        screen_info.append(f'Object labels on: {operating_config.SHOW_DETECT_LABELS}')
        screen_info.append(f'FPS display on:   {operating_config.SHOW_FPS}')
        screen_info.append(f'Confidence Threshold: {operating_config.modelNet.confidence_threshold}')
        screen_info.append(f'NMS Threshold: {operating_config.modelNet.nms_threshold}')
        
        for i in range(1,len(screen_info)+1):
            write_row_of_info(img=img,
                              info=screen_info[i-1],
                              rownum=i,
                              operating_config=operating_config)
