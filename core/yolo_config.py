import cv2 as cv

def get_net_config(model_type, config_dir):
    
    whT, hhT, modelConfiguration, modelWeights = get_model_config(config_dir=config_dir,
                                                                  model_type=model_type)
      
    # Chagned from "FromDarknet" to generic. 2022-09-06
    #net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
    net = cv.dnn.readNet(modelWeights, modelConfiguration)
    
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    #net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    #net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Need to get the names of the output layers.
    # This gives the index of the layers, not the names
    # e.g. value of 200 is 199 (because 0 is a valid layer)
    layerNames = net.getLayerNames()
    print(layerNames)
    print(f'layerNames length: {len(layerNames)}, type: {type(layerNames)}')
    print(net.getUnconnectedOutLayers())

    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    print(outputNames) #(gives output names of the layers)
    
    return net, outputNames, whT, hhT


# Constants to be used for "get_model_config(model_type=)"
MODEL_YOLOV3_320_320=1

MODEL_YOLOV3_320_192=2
MODEL_YOLOV3_416_256=3
MODEL_YOLOV3_576_352=4
MODEL_YOLOV3_608_352=5

MODEL_YOLOV3T_320_192=6
MODEL_YOLOV3T_416_256=7
MODEL_YOLOV3T_576_352=8
MODEL_YOLOV3T_608_352=9

MODEL_YOLOV4N_320_192=10
MODEL_YOLOV4N_416_256=11
MODEL_YOLOV4N_576_352=12
MODEL_YOLOV4N_608_352=13

MODEL_YOLOV4T_320_192=14
MODEL_YOLOV4T_416_256=15
MODEL_YOLOV4T_576_352=16
MODEL_YOLOV4T_608_352=17

def get_model_config(config_dir, model_type):
    
    WEIGHT_YOLOV3='yolov3.weights'
    WEIGHT_YOLOV3T='yolov3-tiny.weights'
    WEIGHT_YOLOV4N='yolov4_new.weights'
    WEIGHT_YOLOV4T='yolov4-tiny.weights'
    
    NET_CONFIG_DIR=f'{config_dir}'
    
    # Config files for yolo models
    CFG_YOLOV3_320_320='yolov3_320_320.cfg'

    CFG_YOLOV3_320_192='yolov3_320_192.cfg'
    CFG_YOLOV3_416_256='yolov3_416_256.cfg'
    CFG_YOLOV3_576_352='yolov3_576_352.cfg'
    CFG_YOLOV3_608_352='yolov3_608_352.cfg'

    CFG_YOLOV3T_320_192='yolov3-tiny_320_192.cfg'
    CFG_YOLOV3T_416_256='yolov3-tiny_416_256.cfg'
    CFG_YOLOV3T_576_352='yolov3-tiny_576_352.cfg'
    CFG_YOLOV3T_608_352='yolov3-tiny_608_352.cfg'

    CFG_YOLOV4N_320_192='yolov4_new_320_192.cfg'
    CFG_YOLOV4N_416_256='yolov4_new_320_192.cfg'
    CFG_YOLOV4N_576_352='yolov4_new_320_192.cfg'
    CFG_YOLOV4N_608_352='yolov4_new_320_192.cfg'

    CFG_YOLOV4T_320_192='yolov4-tiny_320_192.cfg'
    CFG_YOLOV4T_416_256='yolov4-tiny_416_256.cfg'
    CFG_YOLOV4T_576_352='yolov4-tiny_576_352.cfg'
    CFG_YOLOV4T_608_352='yolov4-tiny_608_352.cfg'
    
    
    if model_type == MODEL_YOLOV3_320_320:
        whT = 320
        hhT = 320
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3_320_320}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3}'

    elif model_type == MODEL_YOLOV3_320_192: 
        whT = 320
        hhT = 192
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3_320_192}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3}'

    elif model_type == MODEL_YOLOV3_416_256:
        whT = 416
        hhT = 256
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3_416_256}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3}'

    elif model_type == MODEL_YOLOV3_576_352:
        whT = 576
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3_576_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3}'

    elif model_type == MODEL_YOLOV3_608_352:
        whT = 608
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3_608_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3}'

    elif model_type == MODEL_YOLOV3T_320_192:
        whT = 320
        hhT = 192
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3T_320_192}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3T}'

    elif model_type == MODEL_YOLOV3T_416_256:
        whT = 416
        hhT = 256
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3T_416_256}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3T}'

    elif model_type == MODEL_YOLOV3T_576_352:
        whT = 576
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3T_576_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3T}'

    elif model_type == MODEL_YOLOV3T_608_352:
        whT = 608
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV3T_608_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV3T}'
    
    elif model_type == MODEL_YOLOV4N_320_192:
        whT = 320
        hhT = 192
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4N_320_192}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4N}'

    elif model_type == MODEL_YOLOV4N_416_256:
        whT = 416
        hhT = 256
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4N_416_256}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4N}'

    elif model_type == MODEL_YOLOV4N_576_352:
        whT = 576
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4N_576_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4N}'

    elif model_type == MODEL_YOLOV4N_608_352:
        whT = 608
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4N_608_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4N}'

    elif model_type == MODEL_YOLOV4T_320_192:
        whT = 320
        hhT = 192
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4T_320_192}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4T}'
    
    elif model_type == MODEL_YOLOV4T_416_256:
        whT = 416
        hhT = 256
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4T_416_256}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4T}'

    elif model_type == MODEL_YOLOV4T_576_352:
        whT = 576
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4T_576_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4T}'

    elif model_type == MODEL_YOLOV4T_608_352:
        whT = 608
        hhT = 352
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4T_608_352}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4T}'

    print(f'Returning modelConfiguration: {modelConfiguration}')
    print(f'Returning modelWeights:       {modelWeights}')

    return whT, hhT, modelConfiguration, modelWeights