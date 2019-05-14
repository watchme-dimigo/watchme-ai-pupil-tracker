import cv2
import numpy as np
from preprocessor import (
    get_threshold,
    apply_threshold
)

HAARCASCADE_PATH = '/usr/local/lib/python3.7/site-packages/cv2/data/'

face_cascade = cv2.CascadeClassifier(
    HAARCASCADE_PATH + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(
    HAARCASCADE_PATH + 'haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

while(True):
    _, frame = video_capture.read()
    # frame = cv2.imread('./fig.jpeg')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        face = faces[0]
    except IndexError:
        continue
    (x, y, w, h) = face
    face = gray[y:y+h, x:x+w]
    face_height = np.size(face, 0)

    eyes = eye_cascade.detectMultiScale(face)

    for idx, eye in enumerate(eyes):
        with open('./dataset/normal/index') as index_file:
            index = int(index_file.readline())

        (x, y, w, h) = eye
        if y + h > face_height * 2 / 3: # 내 콧구멍
            continue

        eye = face[y:y+h, x:x+w]

        eye = apply_threshold(eye)

        cv2.imshow('eye-{}'.format(idx), eye)
        cv2.imwrite('dataset/normal/{}.png'.format(idx + index), eye)

        with open('./dataset/normal/index', 'w') as index_file:
            index_file.write(str(idx + index + 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
