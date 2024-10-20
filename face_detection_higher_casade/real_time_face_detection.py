import cv2
import numpy as np
from project_face_detection import face_detection
import os
os.chdir('path/to/directory')

face_cascade=cv2.CascadeClassifier('path/to/model/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if ret == False:
        break
    img_detect= face_detection(frame)
    
    cv2.imshow('Real time Face Detection',img_detect)
    if cv2.waitKey(1)==ord('a'):
        break
    
cap.release()
cv2.destroyAllWindows()
