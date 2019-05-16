import sys
from time import sleep
import cv2
import numpy as np
from preprocessor import (
    get_threshold,
    apply_threshold
)

ALLOWED_CLASSES = ['normal_0', 'top_left_1', 'top_right_2', 'bottom_left_3', 'bottom_right_4']
try:
    CLASS = sys.argv[1]
except IndexError:
    print('No CLASS provided; Should be one of the following: {}'.format(str(ALLOWED_CLASSES)))
    exit(1)
if CLASS not in ALLOWED_CLASSES:
    print('Invalid CLASS; Should be one of the following: {}'.format(str(ALLOWED_CLASSES)))
    exit(1)

try:
    TYPE = sys.argv[2]
    if TYPE not in ['train', 'test']:
        print("Invalid TYPE; Should be one of the following(default 'train'): {}".format(str(['train', 'test'])))
except IndexError:
    TYPE = 'train'

DATASET_PATH = f'./dataset/{TYPE}/{CLASS}/'

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
    if len(faces):
        print('[*] face detected')
    for face in faces:
        (x, y, w, h) = face
        face = gray[y:y+h, x:x+w]
        face_height = np.size(face, 0)

        eyes = eye_cascade.detectMultiScale(face)

        for idx, eye in enumerate(eyes):
            print('################## process eye ##################')

            with open(DATASET_PATH + 'index') as index_file:
                index = int(index_file.readline())

            (x, y, w, h) = eye
            # if y + h > face_height * 2.5 / 3: # 내 콧구멍
            #     continue
            # 아오 그냥 수동으로 지워주자

            eye = face[y:y+h, x:x+w]

            eye = apply_threshold(eye)

            # cv2.imshow('eye-{}'.format(idx), eye)
            cv2.imwrite(DATASET_PATH + CLASS + '_{}.png'.format(idx + index), eye)
            with open(DATASET_PATH + 'index', 'w') as index_file:
                index_file.write(str(idx + index + 1))
            print(DATASET_PATH + '{}.png'.format(idx + index))

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    sleep(1)
