import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
#parameters inside
#static image mode -- will either track or detect based on confidence level. T/F
#minimum tracking and detection confidence -- both by default set to 50%
mpDraw = mp.solutions.drawing_utils
#the frame rate
pTime = 0

while True:

    success, img = cap.read() #simply rendering the web cam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR to RGB conversion for the hand object
    results = hands.process(imgRGB) #process

    #open result up and extract info eg. about multiple hands
    #print(results.multi_hand_landmarks) #to see if there's an object change b/c of hand detection
    h,w, c = img.shape
    if results.multi_hand_landmarks: #process position results
        for handLms in results.multi_hand_landmarks:
            #get the information for each hand now, both id and landmark (x, y coordinate)
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm) #the positio(lm) of each id 0 - 20
                #the location is in dp (x, y, z). but we want it on pixels.
                #multiply by width and height to do that.
                cx, cy = int(lm.x*w), int(lm.y*h) #to pixel from center
                print(id, cx, cy) #each landmark pixel position
                #now let us signify one of the landmarks
                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255, 8, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS) #the BGR 21 landmark hand
    
    #now let us also do the connections
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,3,(255, 8, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)