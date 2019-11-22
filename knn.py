import cv2
import os
import numpy as np
#cap = cv2.VideoCapture(1)
# video_path = 'E:/Fall-Detection/cam5.avi'
video_path = 0
cap = cv2.VideoCapture(video_path)    #调取视频
#cap = cv2.VideoCapture(0)   #调取摄像头

bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)


while True:

    ret, frame = cap.read()
    fgmask = bs.apply(frame)
    th = cv2.threshold(fgmask.copy(),244,255,cv2.THRESH_BINARY)[1]

    dilated = cv2.dilate(
        th,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
        iterations = 2)
    t = 0
    contours,hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    # if(cv2.contourArea(contours[0]) > 1600):
    # #         t+=1
    #     (x,y,w,h) = cv2.boundingRect(contours[0])
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)


    print(t)  #打印当前有几个移动物体
    cv2.imshow('mog', fgmask)
    cv2.imshow('thresh',th)
    cv2.imshow('detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
