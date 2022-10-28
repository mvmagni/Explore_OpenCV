import cv2 as cv
import numpy as np
import os
import utils
import time
import yolo_config as yc
from statistics import mean
from operating_config import operatingConfig
from model_net import ModelNet

main_window_name='ML Image processing'

# Resource configs
PROJECT_ROOT_DIR = f'{os.getcwd()}/Explore_OpenCV'
image_store_dir = f'd:/OBS_Recordings'
model_config_dir = f'{PROJECT_ROOT_DIR}/net_configs'
resource_dir = f'{PROJECT_ROOT_DIR}/resources'
className_file = f'{PROJECT_ROOT_DIR}/net_configs/coco.names'
SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/walk.mp4'
#SAMPLE_VIDEO=f'{PROJECT_ROOT_DIR}/resources/highway.mp4'

#Configs for changing video while running
op_config = operatingConfig(image_store_dir=image_store_dir,
                            resource_dir=resource_dir,
                            model_config_dir=model_config_dir,
                            className_file = className_file)

###############################################################################
# Config video capture. 0 is first
# Added cv.CAP_DSHOW to avoid several minute lag of opening cam on windows
# No lag opening on Linux

# Webcam
cap = cv.VideoCapture(0,cv.CAP_DSHOW)

# Video file defined above
#cap = cv.VideoCapture(SAMPLE_VIDEO)

# Jarvis rtsp stream
#gst = 'rtspsrc location=rtsp://172.20.0.30:8554/unicast latency=10 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
#cap = cv.VideoCapture(gst,cv.CAP_GSTREAMER)

fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,900)
################################################################################

# Load background image
bg_img_orig = cv.imread(f'{op_config.resource_dir}/background.jpg')

while op_config.RUN_PROGRAM: 
    while op_config.CONFIGURE and op_config.RUN_PROGRAM:
        bg_img = bg_img_orig.copy()
        utils.write_config_screen(img=bg_img,
                                  operating_config=op_config)
        cv.imshow(main_window_name, bg_img)
        
        # Handle input for configuration
        k = cv.waitKey(0)
        utils.handle_config_key_input(img=bg_img,
                                      key=k,
                                      operating_config=op_config)
  
    load_bg_img = bg_img_orig.copy()
    utils.write_loading_model(img=load_bg_img,
                              operating_config=op_config)
    cv.imshow(main_window_name, load_bg_img)
    k = cv.waitKey(1)
    
    # Check to see if current detection model matches the desired model
    if not (op_config.detection_model == op_config.modelNet.model_type):
        op_config.create_modelNet()
    else: 
        print(f'Desired detection model already active')

    # Process image
    op_config.PROCESS_IMAGES = True

    while op_config.PROCESS_IMAGES and op_config.RUN_PROGRAM:
        op_config.frame_counter += 1
        success, frame = cap.read()

        if op_config.frame_counter == 1:
            print(f'Image size: {frame.shape}')

        processed_frame = utils.process_image(img=frame,
                                              operating_config=op_config)

        # Show the image
        cv.imshow(main_window_name,processed_frame)

        # Esc to close, space to write a copy of the image
        k = cv.waitKey(1)
        utils.handle_processing_key_input(img=frame,
                                          key=k,
                                          operating_config=op_config)     

# Release the cam link
cap.release()

# Clear all the windows
cv.destroyAllWindows()