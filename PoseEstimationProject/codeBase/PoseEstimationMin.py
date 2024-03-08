import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime = 0
while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark): #id 0 - 32
            h,w,c = img.shape #pixel from center as we have done for the hamd detection
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
    #organise landmarks to be accessible by indexing 0 - 32
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
