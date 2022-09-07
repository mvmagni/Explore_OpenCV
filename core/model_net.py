import cv2 as cv
from dataclasses import dataclass, field
import yolo_config as yc

@dataclass
class ModelNet:
    config_dir: str
    classname_file: str
    model_type: str
    confidence_threshold: float
    nms_threshold: float
    
    target_whT: int = field(init=False)
    target_hhT: int = field(init=False)
    model_config_file: str = field(init=False)
    model_weights_file: str = field(init=False)
    

    def __post_init__(self):
        self.target_whT, self.target_hhT,self.model_config_file, self.model_weights_file = yc.get_model_config(config_dir=self.config_dir,
                                                                             model_type=self.model_type)
        
        net = cv.dnn.readNet(self.model_weights_file,
                             self.model_config_file)
        
        # Enable GPU CUDA
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        self.model = cv.dnn_DetectionModel(net)

        self.classes = get_classNames(self.classname_file)
        self.model.setInputParams(size=(self.target_whT, self.target_hhT), scale=1/255)
        print(f'ModelNet configuration completed')

    def detect(self, img):
        # return example(class_ids, scores, boxes) = detect(img)
        return self.model.detect(img, 
                                 nmsThreshold=self.nms_threshold,
                                 confThreshold=self.confidence_threshold)
   
def get_classNames(classFile):
    # Coco info
    classNames = []

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    return classNames