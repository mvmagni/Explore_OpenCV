import cv2 as cv

# Constants to be used for "get_model_config(model_type=)"
MODEL_YOLOV3_320_320='MODEL_YOLOV3_320_320'

MODEL_YOLOV3_320_192='MODEL_YOLOV3_320_192'
MODEL_YOLOV3_416_256='MODEL_YOLOV3_416_256'
MODEL_YOLOV3_576_352='MODEL_YOLOV3_576_352'
MODEL_YOLOV3_608_352='MODEL_YOLOV3_608_352'

MODEL_YOLOV3T_320_192='MODEL_YOLOV3T_320_192'
MODEL_YOLOV3T_416_256='MODEL_YOLOV3T_416_256'
MODEL_YOLOV3T_576_352='MODEL_YOLOV3T_576_352'
MODEL_YOLOV3T_608_352='MODEL_YOLOV3T_608_352'

MODEL_YOLOV4N_320_192='MODEL_YOLOV4N_320_192'
MODEL_YOLOV4N_416_256='MODEL_YOLOV4N_416_256'
MODEL_YOLOV4N_576_352='MODEL_YOLOV4N_576_352'
MODEL_YOLOV4N_608_352='MODEL_YOLOV4N_608_352'
MODEL_YOLOV4N_800_448='MODEL_YOLOV4N_800_448'
MODEL_YOLOV4N_1248_704='MODEL_YOLOV4N_1248_704'

MODEL_YOLOV4T_320_192='MODEL_YOLOV4T_320_192'
MODEL_YOLOV4T_416_256='MODEL_YOLOV4T_416_256'
MODEL_YOLOV4T_576_352='MODEL_YOLOV4T_576_352'
MODEL_YOLOV4T_608_352='MODEL_YOLOV4T_608_352'
MODEL_YOLOV4T_1248_704='MODEL_YOLOV4T_1248_704'

MODEL_LIST = ['MODEL_YOLOV3_320_320',
              'MODEL_YOLOV3_320_192',
              'MODEL_YOLOV3_416_256',
              'MODEL_YOLOV3_576_352',
              'MODEL_YOLOV3_608_352',

              'MODEL_YOLOV3T_320_192',
              'MODEL_YOLOV3T_416_256',
              'MODEL_YOLOV3T_576_352',
              'MODEL_YOLOV3T_608_352',

              'MODEL_YOLOV4N_320_192',
              'MODEL_YOLOV4N_416_256',
              'MODEL_YOLOV4N_576_352',
              'MODEL_YOLOV4N_608_352',
              'MODEL_YOLOV4N_800_448',
              'MODEL_YOLOV4N_1248_704',

              'MODEL_YOLOV4T_320_192',
              'MODEL_YOLOV4T_416_256',
              'MODEL_YOLOV4T_576_352',
              'MODEL_YOLOV4T_608_352',
              'MODEL_YOLOV4T_1248_704'
              ]

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
    CFG_YOLOV4N_416_256='yolov4_new_416_256.cfg'
    CFG_YOLOV4N_576_352='yolov4_new_576_352.cfg'
    CFG_YOLOV4N_608_352='yolov4_new_608_352.cfg'
    CFG_YOLOV4N_800_448='yolov4_new_800_448.cfg'
    CFG_YOLOV4N_1248_704='yolov4_new_1248_704.cfg'

    CFG_YOLOV4T_320_192='yolov4-tiny_320_192.cfg'
    CFG_YOLOV4T_416_256='yolov4-tiny_416_256.cfg'
    CFG_YOLOV4T_576_352='yolov4-tiny_576_352.cfg'
    CFG_YOLOV4T_608_352='yolov4-tiny_608_352.cfg'
    CFG_YOLOV4T_1248_704='yolov4-tiny_1248_704.cfg'
    
    
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
    elif model_type == MODEL_YOLOV4N_800_448:
        whT = 800
        hhT = 448
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4N_800_448}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4N}'
    elif model_type == MODEL_YOLOV4N_1248_704:
        whT = 1248
        hhT = 704
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4N_1248_704}'
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
    elif model_type == MODEL_YOLOV4T_1248_704:
        whT = 1248
        hhT = 704
        modelConfiguration=f'{NET_CONFIG_DIR}/{CFG_YOLOV4T_1248_704}'
        modelWeights=f'{NET_CONFIG_DIR}/{WEIGHT_YOLOV4T}'


    print(f'Returning modelConfiguration: {modelConfiguration}')
    print(f'Returning modelWeights:       {modelWeights}')

    return whT, hhT, modelConfiguration, modelWeights