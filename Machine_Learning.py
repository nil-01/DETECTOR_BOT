import cv2
from random import randrange

face_detector_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
car_detector_data = cv2.CascadeClassifier('cars.xml')
pedestrial_detector_data = cv2.CascadeClassifier('haarcascade_fullbody.xml')
webcam = cv2.VideoCapture(0) # videocapture you can add file name of mp4 format and image to using 'filename.mp4' or 'filename.png'
while True:

    successful_frame_read,frame=webcam.read()
    if successful_frame_read:
        grayscaled=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    face_coordinates=face_detector_data.detectMultiScale(grayscaled)
    car_coordinates=car_detector_data.detectMultiScale(grayscaled)
    pedestrial_coordinates=pedestrial_detector_data.detectMultiScale(grayscaled)
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(255)),2)
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)
    for (x, y, w, h) in pedestrial_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256),randrange(256)), 2)
    cv2.imshow('MACHINE LEARNING EXAMPLE DEDICATED TO ANGEL',frame)
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break


