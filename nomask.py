# 코와 입 감지 했을때 -> 마스크 안 쓴 사진으로 분류

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

font = cv.FONT_HERSHEY_SIMPLEX
def face_detector():

    face_cascade_file = "cascade/haarcascade_frontalface_alt.xml"
    # mouth_cascade_file = "cascade/haarcascade_eye.xml"
    face_cascade = cv.CascadeClassifier(face_cascade_file)
    # mouth_cascade = cv.CascadeClassifier(mouth_cascade_file)

    cap = cv.VideoCapture(0)
    while True:
        ret, cam = cap.read()

        if ret:
            cv.imshow('camera', cam)

            # press esc to close window
            if cv.waitKey(1) & 0xFF == 27:
                break

        gray = cv.cvtColor(cam, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 3, minSize=(150, 150))
        # mouth = mouth_cascade.detectMultiScale(gray, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv.rectangle(cam, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(cam, "Detected Face", (x - 5, y - 5), font, 0.5, (0, 0, 255), 2)

        # for (x, y, w, h) in mouth:
        #     cv.rectangle(cam, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv.putText(cam, "Detected Eye", (x - 5, y - 5), font, 0.5, (0, 0, 255), 2)

        cv.imshow("cam", cam)
        k = cv.waitKey(30)

    # cap.release()
    cv.destroyAllWindows()

def nomask():
    return

