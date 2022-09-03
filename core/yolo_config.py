import cv2 as cv

def get_net_config(modelConfiguration, 
                   modelWeights):
    
    net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
    
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
    
    return net, outputNames