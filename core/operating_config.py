import cv2 as cv
import os
from dataclasses import dataclass, field

@dataclass
class operatingConfig:
    # Where to store screenshots
    image_store_dir: str
        
    #Configs for changing video while running
    SHOW_FPS: bool = True
    SHOW_DETECT: bool = True
    SHOW_DETECT_LABELS: bool = True
    
    # Runtime variables
    fps_queue: list = field(default_factory=list)
    prev_frame_time: int = 0
    frame_counter: int = 0
    

    def __post_init__(self):
        self.print_environment()

    def print_environment(self):
        print(f'OpenCV Version: {cv.__version__}')
        print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
        print(f'CWD: {os.getcwd()}')