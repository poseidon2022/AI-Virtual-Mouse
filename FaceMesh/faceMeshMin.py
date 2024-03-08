import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
mesh = mpMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1) #mesh circle radiuses personalized and reduced or increased accordingly
pTime = 0 
while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mesh.process(imgRGB)
    h, w, c = img.shape
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face, mpMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            #but in a real and hands on project, u need to know and use this points (the 360 points)
            for id, lm in enumerate(face.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h) #to pixel from center
                print(id, cx, cy) #each landmark pixel position
                #now let us signify one of the landmarks
                if id == 4:
                    cv2.circle(img, (cx,cy), 2, (255, 8, 255), cv2.FILLED)
                
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'FPS: ' + str(int(fps)),(20, 70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)

