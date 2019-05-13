import cv2
from preprocessor import (
    get_threshold,
    apply_threshold
)

HAARCASCADE_PATH = '/usr/local/lib/python3.7/site-packages/cv2/data/'

face_cascade = cv2.CascadeClassifier(
    HAARCASCADE_PATH + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(
    HAARCASCADE_PATH + 'haarcascade_eye.xml')

frame = cv2.imread('./fig.jpeg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
face = faces[0]
(x, y, w, h) = face
face = gray[y:y+h, x:x+w]

eyes = eye_cascade.detectMultiScale(face)
eye = eyes[0]
(x, y, w, h) = eye
eye = face[y:y+h, x:x+w]

eye = apply_threshold(eye)

cv2.imshow('frame', eye)
cv2.waitKey(0)
