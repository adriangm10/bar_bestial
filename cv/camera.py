import sys
from PyQt5 import QtWidgets
QtWidgets.QApplication(sys.argv)

import cv2
import time

def camera_idxs():
    arr = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            arr.append(i)
            cap.release()
    return arr

if __name__ == "__main__":
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    cam_idxs = camera_idxs()

    cam = cv2.VideoCapture(cam_idxs[-1])
    rval = True
    while rval:
        rval, frame = cam.read()
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imshow("preview", frame)

        key = cv2.pollKey()
        if key == 27:
            break
        if key == 13:
            name = f"./opencv_images/opencv_img_{time.time_ns()}.png"
            cv2.imwrite(name, frame)
            print(name, "written")
        elif key != -1:
            print(key)

    cv2.destroyAllWindows()
    cam.release()
