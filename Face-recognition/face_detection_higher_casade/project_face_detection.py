import numpy as np
import cv2
import os
os.chdir('Path/to/your/dir')


def display(window,image,time):
    cv2.imshow(window,image)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def face_detection(image): 
    face_cascade=cv2.CascadeClassifier('path/to/model/haarcascade_frontalface_default.xml')
    
    #1 cnvt image into grayscale
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # apply grayscale img to cascade classifier
    box,detections= face_cascade.detectMultiScale2(gray_image,1.05,9)
    print(box)
    print(detections)
    #step 3 bounding box
    for x,y,w,h in box:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
    return image

if __name__=="__main__":
    img=cv2.imread('faces.jpg')
    #display("faces",img,0)
    image=img.copy()
    face_detection(image)
    display("image",image,0)
    

    
    

   
    
