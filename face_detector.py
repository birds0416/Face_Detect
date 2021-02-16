import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

font = cv.FONT_HERSHEY_SIMPLEX

face_cascade_file = "cascade/haarcascade_frontalface_alt.xml"
# face_cascade_file = "cascade/haarcascade_frontalface_alt.xml"
# eye_cascade_file = "cascade/haarcascade_eye.xml"
face_cascade = cv.CascadeClassifier(face_cascade_file)
# eye_cascade = cv.CascadeClassifier(eye_cascade_file)
# black_mask = cv.imread(file=resource_path("photos/black.png"))
# h_mask, w_mask = black_mask.shape[:2]

cap = cv.VideoCapture(0)
while True:
    ret, cam = cap.read()

    if ret:
        # cv.imshow('camera', cam)

        # press esc to close window
        if cv.waitKey(1) & 0xFF == 27:
            break

    gray = cv.cvtColor(cam, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 3, minSize=(150, 150))
    # mouth = mouth_cascade.detectMultiScale(gray, minSize=(50, 50))

    # if len(faces) == 0:
    #     break

    for (x, y, w, h) in faces:
        cv.rectangle(cam, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(cam, "Detected Face", (x - 5, y - 5), font, 0.5, (0, 0, 255), 2)

        # if h > 0 and w > 0:
        #     x = int(x - w * 0.1)
        #     y = int(y - h * 0.05)
        #     w = int(1.2 * w)
        #     h = int(1.2 * h)
        #
        #     cam_roi = cam[y:y + h, x:x + w]
        #
        #     mask_small = cv.resize(black_mask, (w, h), interpolation=cv.INTER_AREA)
        #     gray_mask = cv.cvtColor(mask_small, cv.COLOR_BGR2GRAY)
        #     ret, mask = cv.threshold(gray_mask, 240, 255, cv.THRESH_BINARY_INV)
        #
        # mask_inv = cv.bitwise_not(mask)
        # masked_face = cv.bitwise_and(mask_small, mask_small, mask=mask)
        # masked_frame = cv.bitwise_and(cam_roi, cam, mask=mask_inv)
        #
        # cam[y:y + h, x:x + w] = cv.add(masked_face, masked_frame)


    cv.imshow("cam", cam)
    k = cv.waitKey(1)

cap.release()
cv.destroyAllWindows()





