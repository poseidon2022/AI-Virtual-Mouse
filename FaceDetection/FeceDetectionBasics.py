import mediapipe as mp
import time
import cv2


mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection(0.75) #default detection confidence was 0.5

cap = cv2.VideoCapture(0)
pTime = 0
while True:

    
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    h, w, c = img.shape
    if results:
        for id, face in enumerate(results.detections):
            #print(id, face)
            print(face)
            bboxC = face.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (0, 255,0), 2) #drawing the bounding_box by ourselves
            cv2.putText(img, str(int(face.score[0] * 100)) + '%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3 )


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, 'FPS: ' + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3 )
    cv2.imshow('Image', img)
    cv2.waitKey(1)


