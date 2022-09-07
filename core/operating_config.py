import cv2 as cv
import os
import yolo_config as yc
from dataclasses import dataclass, field
from model_net import ModelNet

@dataclass
class operatingConfig:
     # Where to store screenshots
    image_store_dir: str
    resource_dir: str
    model_config_dir: str
    className_file: str
    detection_model: str = field(init=False)
      
    # Default model to load
    DEFAULT_MODEL: str = yc.MODEL_YOLOV4N_800_448    
    modelNet: ModelNet = field(init=False)
    
    #Configs for changing video while running
    SHOW_FPS: bool = True
    SHOW_DETECT: bool = True
    SHOW_DETECT_LABELS: bool = True
    SHOW_MODEL_CONFIG: bool = True
    
    # Whether the program should RUN or not
    RUN_PROGRAM: bool = True
    
    # Process video file
    PROCESS_IMAGES: bool = True
    
    # Show Configuration screen
    CONFIGURE: bool = True
    
    # Runtime variables
    fps_queue: list = field(default_factory=list)
    prev_frame_time: int = 0
    frame_counter: int = 0
    
    # How many frames to show Model info on activate/switch
    SHOW_MODEL_DETAILS: bool = True
    SHOW_MODEL_DETAILS_FPS: int = 90
    
    # Confidence threshold adjustment amount used in interface
    CONFIDENCE_THRESHOLD=0.6
    CONF_THRESHOLD_ADJUSTBY=0.05
    NMS_THRESHOLD=0.3
    NMS_THRESHOLD_ADJUSTBY=0.05
    
    # Default common values
    font = cv.FONT_HERSHEY_SIMPLEX

    def __post_init__(self):
        self.detection_model = self.DEFAULT_MODEL
        self.create_modelNet()
        self.print_environment()

    def print_environment(self):
        print(f'OpenCV Version: {cv.__version__}')
        print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
        print(f'CWD: {os.getcwd()}')
        
    def create_modelNet(self):
        self.modelNet= ModelNet(config_dir=self.model_config_dir,
                                classname_file=self.className_file,
                                model_type=self.detection_model,
                                confidence_threshold=self.CONFIDENCE_THRESHOLD,
                                nms_threshold=self.NMS_THRESHOLD
                                )