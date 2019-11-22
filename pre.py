import os
import cv2
import csv
import time
import pandas as pd
# import kalman
import numpy as np
# import minpy.numpy as np
import matplotlib.pyplot as plt

# plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
x = []
y = []

def get_data_from_video(path):
    # draw the picture
    cap = cv2.VideoCapture(path)
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, sz)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # cap.set(cv2.CAP_PROP_FPS, 30)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frameNum = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if (cap.get())
        if ret == True:
            frameNum += 1
            frame = cv2.resize(frame, (320,240))
            tempframe = frame.copy()
            if (frameNum == 1):
                previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
                # print("Origin")
            if (frameNum >= 2):
                currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
                currentframe = cv2.absdiff(currentframe, previousframe)
                ret, currentframe = cv2.threshold(currentframe, 16, 255, cv2.THRESH_BINARY)
                # currentframe = cv2.dilate(currentframe, None, iterations = 3)
                # currentframe = cv2.erode(currentframe, None, iterations = 3)
                currentframe = cv2.GaussianBlur(currentframe, (5, 5), 0)

                cnts, hierarchy = cv2.findContours(currentframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    y.append(cv2.contourArea(c))
                    x.append(frameNum)

                    # plt.pause(0)
                    # print(y)
                # for c in cnts:
                #     if cv2.contourArea(c) < 1000 or cv2.contourArea(c) > 19200:
                #         continue
                #     # c是一个二值图，boundingRect是矩形边框函数，用一个最小的矩形，把找到的形状包起来；
                #     # x,y是矩形左上点的坐标；w,h是矩阵的宽和高
                #     (x,y,w,h) = cv2.boundingRect(c) # update the rectangle
                #     hull = cv2.convexHull(c)
                #     # cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
                #     area = cv2.contourArea(hull)

                # rectangle画出矩形，frame是原图，(x,y)是矩阵的左上点坐标，(x+w,y+h)是矩阵右下点坐标
                # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                # Display the resulting frame
                out.write(currentframe)
                cv2.imshow('frame', frame)
                # cv2.imshow('threshold', threshold_frame)
                cv2.imshow('gauss_image', currentframe)
                # cv2.imshow('currentframe', currentframe)
                # write the flipped frame
                # cv2.imshow('gauss', gauss_image)
                # Press Q on keyboard to  exit
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
            previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    return True

if __name__ == '__main__':
    video_path = 'E:/Fall-Detection/cam5.avi'
    # video_path = 1
    # for i in range(2,5):
    #     get_data_from_video('E:/Fall-Detection/cam{}.avi'.format(i))

    get_data_from_video('E:/Fall-Detection/cam4.avi')
    plt.plot(x,y)
    plt.show()
    plt.pause(0)


### 整理，参考详细说明书，专利事宜
