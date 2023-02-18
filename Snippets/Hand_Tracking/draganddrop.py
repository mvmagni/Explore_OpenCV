import cv2
from HandTrackingModule import HandDetector
import numpy as np
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,900)

detector = HandDetector(min_detection_confidence=0.8)

# Initial color for square
colorR = (255, 0, 255)

# Square information
cx, cy, w, h = 100, 100, 200, 200


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    detectedHands, img = detector.findHands(img)
    
    
    if detectedHands:
        for idx, x in enumerate(detectedHands):
            #print(idx,x)
            print(len(detectedHands))
        #if lmList[0] is not None: 
        lmList = detectedHands[0].get('lmList') #Only using first hand detected
        
        cursorX, cursorY, cursorZ = lmList[8]
        dist_8_12, _, _ = detector.findDistance(lmList[8], lmList[12], img)
        print (f'8_12 distance: {dist_8_12}')
        
        #print(f'X:{cursorX}, Y:{cursorY}, Z:{cursorZ}')
        #print(cursor.keys())
        
        if dist_8_12 < 70: #click
            if cx-w//2 < cursorX < cx+w//2 and cy-h//2 < cursorY < cy+h//2:
                colorR = (0, 255, 0)
                cx, cy = cursorX, cursorY
        else:
                colorR = (255, 0, 255)
 
    
    # Draw solid
    #cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
    
    # Draw transparent
    imgNew = np.zeros_like(img, np.uint8)
    cv2.rectangle(imgNew, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    
    
    
    cv2.imshow("Image", out)
    cv2.waitKey(1)