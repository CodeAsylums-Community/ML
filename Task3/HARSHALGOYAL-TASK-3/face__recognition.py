#importing necessary modules
import numpy as np
import cv2

#traning the cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#creating a instanace to capture the video from webcam
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

#calling a loop till the esc key is pressed
while True:
    ret, frame = cap.read() #storing the captured frame in 'frame'  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #converting the captured frame to grayscale
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    #for drawing the rectangle on face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  

    #for displaying the output to screen
    cv2.imshow('video',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

#terminating the program and destroying the created window
cap.release()
cv2.destroyAllWindows()