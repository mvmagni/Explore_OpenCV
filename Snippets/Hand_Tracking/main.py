import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)    
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks: # if at least 1 hand detected
        for handLms in results.multi_hand_landmarks: # loop through each detected hand
            for id, lm in enumerate(handLms.landmark): # get the id and landmark for each point on the current hand
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(f'id: {id} cx: {cx}, cy: {cy}')
                if id == 0: #id 0 is base of hand near wrist
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
                
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
    
    cv2.imshow('image', img)
    cv2.waitKey(1)
