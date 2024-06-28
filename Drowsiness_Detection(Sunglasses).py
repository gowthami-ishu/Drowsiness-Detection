import time

import cv2
import dlib
import imutils
import numpy as np
import pygame
from imutils import face_utils
from scipy.spatial import distance

# Initialize Pygame and load sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("./res/beep-warning-6387.mp3")

# Load pre-trained face landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./res/shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio (EAR) threshold and consecutive frame count for alerts
EAR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

# Initialize variables
alert_status = False
consecutive_closed_frames = 0


# Eye aspect ratio calculation function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Main loop for drowsiness detection
cap = cv2.VideoCapture(0)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EAR_THRESH:
            consecutive_closed_frames += 1
            if consecutive_closed_frames >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(
                    frame,
                    "****************ALERT!****************",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "****************ALERT!****************",
                    (10, 325),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                if not pygame.mixer.get_busy():
                    alert_sound.play()
        else:
            consecutive_closed_frames = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
