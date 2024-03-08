import time
import cv2
import mediapipe as mp

class FaceMesh():

    def __init__(self, static_image_mode=False, max_num_faces=2,
               refine_landmarks=False, min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.mesh = self.mpMesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                               self.refine_landmarks, self.min_detection_confidence,
                               self.min_tracking_confidence)
    
    def findMesh(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.mesh.process(imgRGB)
        drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)
        if self.results.multi_face_landmarks:
            for face in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, face , self.mpMesh.FACEMESH_CONTOURS, drawSpec, drawSpec) 
            
        return img

    def findPosition(self, img, faceNo = 0, draw = True):

        lmlist = []
        h,w,c = img.shape
        if self.results.multi_face_landmarks:
            face = self.results.multi_face_landmarks[faceNo]
            for id, lm in enumerate(face.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h) #to pixel from center
                lmlist.append([id, cx, cy]) #each landmark pixel position
                #now let us signify one of the landmarks
                if draw:
                    cv2.circle(img, (cx,cy), 2, (255, 8, 255), cv2.FILLED)
        
        return lmlist


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        mine = FaceMesh()
        img = mine.findMesh(img)
        lmlist = mine.findPosition(img)
        if lmlist:
            print(lmlist[38])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img,'FPS: ' + str(int(fps)), (70, 20),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)
if __name__ == '__main__':
    main()