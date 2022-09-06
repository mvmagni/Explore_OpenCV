import cv2 as cv

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

# Speed higher than CRST but less accurate
#tracker = cv.legacy.TrackerMOSSE_create()

# slower but more accurate than above
tracker = cv.legacy.TrackerCSRT_create()

success, img = cap.read()

# Uses mouse cursor to select Region of Interest
bbox = cv.selectROI('Tracking', img, False)
tracker.init(img,bbox)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(img, 
                 (x,y),
                 ((x+w), y+h),
                 (255,0,255),
                 3, # Thickness
                 1) # Line type
    
    cv.putText(img, 
        "Tracking...", 
        (15, 75), 
        cv.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 255, 0),
        2, 
        cv.LINE_AA) 
    


while True:
    timer = cv.getTickCount()
    success, img = cap.read()
    
    success,bbox = tracker.update(img)
    
    if success:
        drawBox(img, bbox)
    else:
        cv.putText(img, 
                "Tracking lost", 
                (15, 75), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.7, (100, 255, 0),
                2, 
                cv.LINE_AA)    
            
    
    fps = round(cv.getTickFrequency()/(cv.getTickCount() - timer))
    cv.putText(img, 
               str(fps), 
               (15, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 
               0.7, (100, 255, 0),
               2, 
               cv.LINE_AA)
    cv.imshow('OpenCV Test', img)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break