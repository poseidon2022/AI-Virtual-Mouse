import mediapipe as mp
import cv2
import time
import math
class handDetector():

    def __init__(self, mode = False,  max_num_hands = 2, model_com = 1,detection_confidence = 0.5, track_confidence = 0.5):
        
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.model_com = model_com

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_num_hands, self.model_com,
                                        self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR to RGB conversion for the hand object
        self.results = self.hands.process(imgRGB) #process
        #open result up and extract info eg. about multiple hands
        #print(results.multi_hand_landmarks) #to see if there's an object change b/c of hand detection
        if self.results.multi_hand_landmarks: #process position results
            for handLms in self.results.multi_hand_landmarks:
                #get the information for each hand now, both id and landmark (x, y coordinate)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS) #the BGR 21 landmark hand

                
        return img #we might need to draw on it that's why
    
    def findPosition(self, img, handNo = 0, draw = True):
        self.lmList = []
        xList = []
        yList = []
        h,w, c = img.shape
        if self.results.multi_hand_landmarks: 
            myHand = self.results.multi_hand_landmarks[handNo]
            #will get the first hand and enumerate all the land marks within the hand
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm) #the positio(lm) of each id 0 - 20
                #the location is in dp (x, y, z). but we want it on pixels.
                #multiply by width and height to do that.
                cx, cy = int(lm.x*w), int(lm.y*h) #to pixe from center
                xList.append(cx)
                yList.append(cy)
                #each landmark pixel position
                self.lmList.append([id, cx, cy])
                #now let us signify one of the landmarks
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255, 8, 255), cv2.FILLED)
            
            xMax, xMin = max(xList), min(xList)
            yMax, yMin = max(yList), min(yList)
            if draw:
                cv2.rectangle(img, (xMin - 20, yMin - 20), (xMax + 20, yMax + 200), (0,255,0), 2)

            
        return self.lmList

    
    def findIfUp(self):

        fingers = []
        tipIDs = [4,8,12,16,20]
        #for the thumb, something is weird
        if self.lmlist[4][1] < self.lmlist[3][1]:
            fingers.append(0)
        else:
            fingers.append(1)
        for id in range(1,5):

            if self.lmlist[tipIDs[id]][2] > self.lmlist[tipIDs[id - 2]][2]:
                fingers.append(0)
            else:
                fingers.append(1)
            
        return fingers

    def findDistance(self, p1, p2, img, draw = True, t = 3, r = 15):

        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]

        cx, cy = (x1 + x2) // 2, (y1 + y2)//2

        if draw:

            cv2.line(img, (x1,y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255),cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255,0,255),cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0,0,255),cv2.FILLED)
        
        length = math.hypot((x2 - x1), (y2 - y1))

        return length, img, [x1,y1,x2,y2,cx,cy]
            
def main():
    
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:

        success, img = cap.read() #simply rendering the web cam
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if lmList:
            print(lmList[5])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,3,(255, 8, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)






if __name__=="__main__":
    main()