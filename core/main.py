import cv2 as cv
import numpy as np
import os
import utils
import yolo_config as yc
from statistics import mean
from operating_config import operatingConfig
from model_net import ModelNet

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
#cap = cv.VideoCapture(0,cv.CAP_DSHOW)
cap = cv.VideoCapture(SAMPLE_VIDEO)
fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)
################################################################################

# Initialize empty ModelNet as mn
mn=None
while op_config.RUN_PROGRAM:
    if mn is None:
        # Create a default model
        mn = ModelNet(config_dir=op_config.model_config_dir,
                      classname_file=op_config.className_file,
                      model_type=op_config.detection_model
                      )
    
    while op_config.CONFIGURE:
        # Load image
        bg_img = cv.imread(f'{op_config.resource_dir}/background.jpg')
        
        utils.write_config_screen(img=bg_img,
                                  modelNet=mn,
                                  operating_config=op_config)
        cv.imshow("OpenCV Test", bg_img)
        
        # Handle input for configuration
        k = cv.waitKey(5)
        utils.handle_config_key_input(img=bg_img,
                                      key=k,
                                      operating_config=op_config)


    # Check to see if current detection model matches the desired model
    if not (op_config.detection_model == mn.model_type):
        mn = ModelNet(config_dir=op_config.model_config_dir,
                      classname_file=op_config.className_file,
                      model_type=op_config.detection_model
                      )
    else: 
        print(f'Desired detection model already active')

    while op_config.PROCESS_IMAGES:
        op_config.frame_counter += 1
        success, frame = cap.read()

        if op_config.frame_counter == 1:
            print(f'Image size: {frame.shape}')

        utils.process_image(modelNet=mn,
                            img=frame,
                            operating_config=op_config)

        # Show the image
        cv.imshow('OpenCV Test',frame)

        # Esc to close, space to write a copy of the image
        k = cv.waitKey(1)
        utils.handle_processing_key_input(img=frame,
                                          key=k,
                                          operating_config=op_config)     


# Release the cam link
cap.release()

# Clear all the windows
cv.destroyAllWindows()