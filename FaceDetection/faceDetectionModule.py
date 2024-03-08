import cv2
import time
import mediapipe as mp

class FaceDetection():

    def __init__(self, min_detection_confidence=0.5, model_selection=0):

        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mpFace = mp.solutions.face_detection
        self.faceDetection = self.mpFace.FaceDetection(self.min_detection_confidence,
                                                       self.model_selection)

    def findFace(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        h, w, c  = img.shape
        if self.results:

            for id, face in enumerate(self.results.detections):
                #print(id, face)
                print(face)
                bboxC = face.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                        int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(img, bbox, (0, 255,0), 2) #drawing the bounding_box by ourselves
                cv2.putText(img, str(int(face.score[0] * 100)) + '%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3 )


        return img
def main():

    cap = cv2.VideoCapture(0)
    mine = FaceDetection()
    pTime = 0
    while True:

        success, img = cap.read()
        img = mine.findFace(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, 'FPS: ' + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0))
        cv2.imshow('Image', img)
        cv2.waitKey(1)
if __name__=='__main__':
    main()