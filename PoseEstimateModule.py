import cv2
import mediapipe as mp
import time


class PoseDetector():

    def __init__(self, mode=False, model_complexity = 1, smooth_landmarks = True,
                 enable_segmentation = False, smooth_segmentation = True,
                 detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detection_confidence, self.tracking_confidence)

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS,
                                      self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                              circle_radius=1),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 0),
                                                                   thickness=2, circle_radius=1))
        return img

    def getPosition(self, img, draw = True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, land_mark in enumerate(self.results.pose_landmarks.landmark):
                height, weight, channel = img.shape
                cx, cy = int(land_mark.x * weight), int(land_mark.y * height)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (0, 0, 255), cv2.FILLED)

        return  lm_list




def main():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    current_time = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()

        img = detector.findPose(img)
        lm_list = detector.getPosition(img)
        print(lm_list)

        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()