# Import necessary libraries
import cv2
import PoseEstimationModule as pem  # Custom module for pose detection
import time
import mediapipe as mp
import os
import numpy as np

# Load background images for different exercises from the given folder
path = "New Images"
Images = os.listdir(path)
print(Images)

# Store the images in a list
List = []
for i in Images:
    im = cv2.imread(path + '/' + i)
    List.append(im)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Set the initial background image
bg_img = List[0]

# Initialize pose detector
detector = pem.poseDetector()

# Initialize exercise counters
jump_cnt, pushups_cnt, twister_cnt, jj_cnt, sq_cnt, dumb_cnt = 0, 0, 0, 0, 0, 0

# Delay counters to avoid multiple counts for the same movement
delay_counter = 0
delay_counter_2 = 0

# Initialize visual progress bar parameters
bar, per = 700, 0

# Start real-time video capture loop
while True:
    success, img = cap.read()

    # Make a copy of the frame to use for pose estimation
    img2 = img.copy()
    img2 = cv2.resize(img2, (1000, 700))
    img2 = cv2.flip(img2, 1)  # Mirror the image for natural interaction
    img = cv2.resize(img, (1500, 800))  # Resize main image

    # Detect pose and landmarks
    img2 = detector.findPose(img2, draw=False)
    lmList = detector.getPosition(img2, draw=False)
    print(lmList)  # Debugging output for landmark positions

    st = ""

    if lmList:
        # Calculate angle at left elbow
        angle = detector.findAngle(img2, 11, 13, 15, draw=False)

        # Extract important landmarks for motion analysis
        left_hand, right_hand = lmList[18][1:], lmList[17][1:]
        left_leg, right_leg = lmList[28][1:], lmList[27][1:]
        left_hip, right_hip = lmList[24][1:], lmList[23][1:]
        left_leg_knee, right_leg_knee = lmList[26][1:], lmList[25][1:]
        left_hand_knee, right_hand_knee = lmList[14][1:], lmList[13][1:]

        # Calculate distances between key points
        crunch_dist1, _, _ = detector.findDistance(left_hand_knee, right_leg_knee)
        crunch_dist2, _, _ = detector.findDistance(right_hand_knee, left_leg_knee)
        cross_dist1, _, _ = detector.findDistance(left_hand, right_leg)
        cross_dist2, _, _ = detector.findDistance(right_hand, right_leg)
        hand_dist, _, _ = detector.findDistance(left_hand, right_hand)
        eye = lmList[1][1:]  # Eye landmark to detect vertical motion

        # Handle fast-action movements based on eye and hand positions
        if delay_counter == 0:
            if eye[1] < 70:  # Likely a jump
                bg_img = List[2]
                per = 100
                bar = 200
                jump_cnt += 1
            elif left_hand[0] > right_hip[0] and left_hand[1] < right_hip[1]:  # Twister movement
                dist = left_hand[1] < right_hip[1]
                per = 100
                bar = 200
                twister_cnt += 1
                bg_img = List[4]
            elif left_hand[0] < 300 and right_hand[0] > 650:  # Jumping jacks motion
                per = 100
                bar = 200
                jj_cnt += 1
                bg_img = List[3]

        # Handle slower movements using angle and height calculations
        if delay_counter_2 == 0:
            per = np.interp(angle, (240, 320), (0, 100))
            bar = np.interp(angle, (240, 320), (700, 200))

            # Pushups detection based on elbow angle and head position
            if eye[1] > 400:
                per = np.interp(angle, (200, 250), (0, 100))
                bar = np.interp(angle, (200, 250), (700, 200))
                if angle > 250:
                    pushups_cnt += 1
                    bg_img = List[5]

            # Squats detection using distance between hip and knee
            elif eye[1] < 400 and abs(left_hip[1] - left_leg_knee[1]) <= 60:
                dist = -abs(left_hip[1] - left_leg_knee[1])
                per = np.interp(dist, (-100, -60), (0, 100))
                bar = np.interp(dist, (-100, -60), (700, 200))
                sq_cnt += 1
                bg_img = List[6]

            # Dumbbell curls based on elbow extension angle
            elif angle >= 320:
                dumb_cnt += 1
                bg_img = List[1]

    # Update delay counters to avoid multiple detections
    delay_counter += 1
    delay_counter_2 += 1
    if delay_counter == 9:
        delay_counter = 0
    if delay_counter_2 == 15:
        delay_counter_2 = 0

    # Overlay background image and draw main webcam feed
    bg_img = cv2.resize(bg_img, (1500, 800))
    img[:800, :1500] = bg_img
    img[100:800, 250:1250] = img2

    # Draw progress bar
    cv2.rectangle(img, (1100, 200), (1200, 700), (0, 255, 0), 4)
    cv2.rectangle(img, (1100, int(bar)), (1200, 700), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, str(int(per)) + "%", (1100, 170), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display counters for different exercises
    cv2.putText(img, str(jj_cnt), (30, 730), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)
    cv2.putText(img, str(jump_cnt), (30, 480), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)
    cv2.putText(img, str(dumb_cnt), (30, 260), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)
    cv2.putText(img, str(twister_cnt), (1360, 260), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)
    cv2.putText(img, str(pushups_cnt), (1360, 480), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)
    cv2.putText(img, str(sq_cnt), (1360, 730), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)

    # Display the final image
    cv2.imshow("My AI Trainer", img)
    cv2.waitKey(1)
