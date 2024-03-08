import handtracikngmodule as htm
import cv2
import mediapipe as mp
import time
import numpy as np
import math
import pycaw as pc
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480 #height and width of our camera
detector = htm.handDetector(detection_confidence=0.7)
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam) #prop ids number 3 and 4 respectively for the w and height
pTime = 0


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange() #max and min range is found to be -74 and 0 # -20, the volume goes to 26
#-5 72, 0 - 100
#this says the fact that, when decreasing to 0, the volume goes to 100
min_volume = volumeRange[0]
max_volume = volumeRange[1]
#converting the volume ranges to the length, our hand range was from 300 to 50 to our
#volume range i.e. -74 to 0
#we have a numpy function to do that




while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw = False)
    if lmlist:
        cx, cy = (lmlist[4][1] + lmlist[8][1])//2, (lmlist[4][2] + lmlist[8][2]) // 2
        x1,x2 = lmlist[4][1], lmlist[8][1]
        y1, y2 = lmlist[4][2],lmlist[8][2]
        #now let us find the centerline between this two points
        cv2.circle(img, (lmlist[4][1],lmlist[4][2]), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (lmlist[8][1],lmlist[8][2]), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (lmlist[4][1],lmlist[4][2]), (lmlist[8][1],lmlist[8][2]), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
    #this draws a line between the given two points.
    #but one and most important thing we should know is the length of the line connecting the two
        length = math.hypot(x2 - x1, y2 - y1)
        print(length)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        
        vol = np.interp(length, [20,200], [min_volume, max_volume]) #the mapping is actually done here
        volume.SetMasterVolumeLevel(vol, None)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, 'FPS: ' + str(int(fps)), (40, 50), cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)


