import mediapipe as mp
import cv2
import time
import numpy as np
import handtracikngmodule as htm
import autopy

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
hand = htm.handDetector(max_num_hands=1)
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hScr = autopy.screen.size() #the actual size of my computer screen
frameR = 100
smoothening = 7
pLocX, pLocY = 0,0
cLocX, cLocY = 0,0
#1536.0, 864.0
pTime = 0
while True:
    #only index finger is up ==> moving mode
    #both the index and middle finger are up and their distance is less than some value ==> clicking mode
    #converting the window display size to our screen size., get the actual coordinate
    success, img = cap.read()
    img = hand.findHands(img)
    lmlist = hand.findPosition(img)

    if lmlist:
        index_x, index_y = lmlist[8][1:]
        middle_y, middle_x = lmlist[12][1:]

        fingers = hand.findIfUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0,255,0),2)
        if fingers[1] == 1 and fingers[2]==0: #index up and middle down
            converted_index_x = np.interp(index_x, (frameR, wCam - frameR), (0, wScr))
            converted_index_y = np.interp(index_y, (frameR, hCam - frameR), (0, hScr))

            cLocX  = pLocX + (converted_index_x - pLocX) / smoothening
            cLocY  = pLocY + (converted_index_y - pLocY) / smoothening
            autopy.mouse.move(wScr - cLocX, cLocY)
            #now we got the coordinates, we should now send this to the mouse
            cv2.circle(img, (index_x, index_y),15,(255,0,0), cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY
        if fingers[1] and fingers[2]:
            length, i, lineInfo = hand.findDistance(8, 12, img) #finding the distance between index and middle
            #if the distance between these two fingers is below some value we'll click.
            if length  < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),15,(0,255,0), cv2.FILLED) #draw a circle at the mid pt
                autopy.mouse.click()
                #but our click and point indication is shaking alot. we have to smoothen the values
            print(length)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'FPS: ' + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(255, 0, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
