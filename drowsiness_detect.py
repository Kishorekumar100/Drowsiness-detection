from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2
import os
import sys

# ---------------- Configuration ----------------
ALERT_AUDIO = "audio/alert.wav"
DLIB_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 50
CAMERA_INDEX = 0
# ------------------------------------------------

# Initialize pygame for alert sound
pygame.mixer.init()
if os.path.exists(ALERT_AUDIO):
    pygame.mixer.music.load(ALERT_AUDIO)

# Eye Aspect Ratio function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

# Check predictor model
if not os.path.exists(DLIB_PREDICTOR):
    print(f"[ERROR] Missing predictor file: {DLIB_PREDICTOR}")
    sys.exit(1)

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PREDICTOR)

# Eye landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start camera
video_capture = cv2.VideoCapture(CAMERA_INDEX)
if not video_capture.isOpened():
    print(f"[ERROR] Cannot access camera index {CAMERA_INDEX}")
    sys.exit(1)

time.sleep(1.0)  # warmup
COUNTER = 0

print("[INFO] Starting drowsiness detection. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None:
        print("[WARN] Skipping empty frame...")
        continue

    # Force frame into uint8 contiguous array
    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    # Convert to grayscale (dlib prefers this)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    # Try detecting faces safely
    try:
        faces = detector(gray, 0)
    except Exception as e:
        print(f"[WARN] Detector error: {e}")
        faces = []

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Drowsiness logic
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if os.path.exists(ALERT_AUDIO) and not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show video feed
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
