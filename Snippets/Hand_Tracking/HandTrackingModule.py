import cv2
import mediapipe as mp
import time 

class handDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=4,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.static_image_mode=static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.static_image_mode,
                      max_num_hands=self.max_num_hands,
                      min_detection_confidence=self.min_detection_confidence,
                      min_tracking_confidence=self.min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
    
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)    
        #print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks: # if at least 1 hand detected
            for handLms in results.multi_hand_landmarks: # loop through each detected hand
                if draw: 
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    

        
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)    
        
        if results.multi_hand_landmarks: # if at least 1 hand detected
            myHand = results.multi_hand_landmarks[handNo]
            
            
            for id, lm in enumerate(myHand.landmark): # get the id and landmark for each point on the current hand
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(f'id: {id} cx: {cx}, cy: {cy}')
                lmList.append([id,cx,cy])
                #if id == 0: #id 0 is base of hand near wrist
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,900)
    detector = handDetector()
    
    while True:
        success, img = cap.read()   
        img = detector.findHands(img=img)
        lmList = detector.findPosition(img=img, handNo=0, draw=False)
        if len(lmList) != 0:
            print(lmList[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        #cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
        
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()