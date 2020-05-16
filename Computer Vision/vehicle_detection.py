##### 1. Using unsupervised methods ####
import pandas as pd
import cv2
import numpy as np
import random
import os

os.chdir(r"C:\Users\schoud790\Documents\Python Codes\computer vision")

video_file = "Traffic - 20581.mp4"

#if the input is from the camera, pass 0 instead of filename
video = cv2.VideoCapture(video_file)
success, prev_frame = video.read()

frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = video.get(cv2.CAP_PROP_FPS)

while success: #loop through the frames
    success, next_frame = video.read()

    if not success:
        break

    #convert to grey scale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    #blur the image to reduce any noise
    prev_gray = cv2.GaussianBlur(prev_gray, (5,5),0)
    next_gray = cv2.GaussianBlur(next_gray, (5,5),0)

    #take frame difference
    diff = cv2.absdiff(next_gray, prev_gray)

    #binarize the frame difference
    #t, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    #dilate the image to imrove the moving object connection
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel=np.ones((5,5)), iterations=2)

    #define a vehicle detection zone
    #assume the zone to be a horizontal line passing from the middle of the frame
    x1, x2 = np.int(binary.shape[1]*0.2), np.int(binary.shape[1]*0.8)
    y1 = y2 = np.int(binary.shape[0]*0.5)

    height = binary.shape[0]
    width = binary.shape[1]


    #find contours
    contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #filter the contours from the detection zone and remove other noise contours
    valid_contours = []
    for i, cntr in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cntr) #get the bounding box

        #check if box lies in the zone and satisfies the minimum size constraint
        if (x2 >= x >= x1) and y <= y1 <= y+h and (cv2.contourArea(cntr) >= width*height*0.01):
            valid_contours.append([(x,y), (x+w, y+h)])
        #else:
        #    valid_contours.append([(x,y), (x+w, y+h)])
    
    #draw the contours
    next_frame_cpy = next_frame.copy() 
    next_frame_cpy = cv2.line(next_frame_cpy, (x1,y1), (x2,y2), color=(0,0,100), thickness=2) #zone
    for p1, p2 in valid_contours:
        #r,g,b = np.int(random.random()*255), np.int(random.random()*255), np.int(random.random()*255)
        b,g,r=100, 0, 0
        next_frame_cpy = cv2.rectangle(next_frame_cpy, p1, p2, color=(b,g,r), thickness=2)

    #count vehicles
    font = cv2.FONT_HERSHEY_SIMPLEX
    next_frame_cpy = cv2.putText(next_frame_cpy, "Vehicles Detected: " + str(len(valid_contours)), (55, 15), font, 0.6, (0, 180, 0), 2)


    cv2.imshow("Vehicles", next_frame_cpy)
    cv2.waitKey(125)

    prev_frame = next_frame

    if 0xFF == ord('q'): #stop if q is pressed
        break

cv2.destroyAllWindows()

def display_video(video):

cv2.imshow("a", binary)
cv2.waitKey()


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
frame_width = binary.shape[1]
frame_height = binary.shape[0]
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
cap = cv2.VideoCapture(video_file)
while cap.isOpened():
    ret, frame = cap.read()
    out.write(frame)
    if ret==True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(125) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

cv2.boundingRect(contours[8])
x2

for i, cntr in enumerate(contours):
    print(i, cv2.contourArea(cntr))
    x,y,w,h = cv2.boundingRect(cntr) #get the bounding box




##### 2. Using unsupervised methods ####
####### DL ########
