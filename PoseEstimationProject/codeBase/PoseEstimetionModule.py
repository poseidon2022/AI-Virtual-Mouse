import mediapipe as mp
import time
import cv2


class PoseEstimation():

    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity,
                                      self.smooth_landmarks,self.enable_segmentation,
                                      self.smooth_segmentation, self.min_detection_confidence,
                                      self.min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils
    
    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            
        return img

    def findPosition(self, img, draw = True):

        lm_list = []
        h,w,c = img.shape
        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark): #id 0 - 32
                
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255, 8, 255), cv2.FILLED) #daw on all parts
        
        return lm_list

def main():
    cap = cap = cv2.VideoCapture(0)
    estimator = PoseEstimation()
    pTime = 0
    while True:
        success, img = cap.read() #simply rendering the web cam
        img = estimator.findPose(img)
        lm_list = estimator.findPosition(img, draw = False)

        cv2.circle(img, (lm_list[5][1],lm_list[5][2]), 5, (255, 8, 255), cv2.FILLED) #only the ears are findposed and drawn now
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,3,(255, 8, 255), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()

            