from datetime import datetime
import yolo_config as yc

startModel = yc.MODEL_YOLOV4T_608_352

lst = yc.MODEL_LIST
print(lst)

print(lst.index(startModel))

increment = 1


def printIdx(increment, lst, currModel):
    print(f'Length of list: {len(lst)}')
    print(f'currModel:      {currModel}')

    desired_model=None
    
    curr_model_index=lst.index(currModel)    
    
    # deal with edge cases first
    if curr_model_index == (len(lst)-1) and increment == 1: # Case end of list
        print(f'Current index at end of list')
        desired_model=lst[0]
    elif curr_model_index == 0 and increment == -1: # Case start of list
        print(f'Current index at start of list')
        desired_model=lst[len(lst)-1]
    else:
        print(f'Standard case: increment by: {increment}')
        desired_model=lst[curr_model_index+increment]
    
    print(f'Desired model: {desired_model}')
 

printIdx(increment=1,
         lst=lst,
         currModel=startModel)

printIdx(increment=-1,
         lst=lst,
         currModel=startModel)




