import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

previous_time = 0
current_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS,
                              mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                              mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1))

        for id, land_mark in enumerate(results.pose_landmarks.landmark):
            height, weight, channel = img.shape
            print(id, land_mark)
            cx, cy = int(land_mark.x * weight), int(land_mark.y * height)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), cv2.FILLED)


    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)