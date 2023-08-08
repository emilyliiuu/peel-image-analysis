# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:56:00 2023

@author: 3mai1
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

ef10 = r"C:\Users\3mai1\Downloads\EF10Front-06152023124024-0000_106kPa.mp4"
ef20 = r"C:\Users\3mai1\Downloads\EF20Front-06152023130308-0000_112kPa.mp4"
efgel = r"C:\Users\3mai1\Downloads\EFGelFront-06152023121715-0000_37kPa.mp4"
ms30 = r"C:\Users\3mai1\Downloads\MS30Front-06152023140708-0000_663kPa.mp4"
ef30 = r"C:\Users\3mai1\Downloads\EF30Front-06152023132505-0000_180kPa.mp4"
ef50 = r"C:\Users\3mai1\Downloads\EF50Front-06152023134551-0000_240kPa.mp4"
substrates = [ms30, ef50, ef30, ef20, ef10, efgel]
distances = []
frames = []
f = 0

cap = cv2.VideoCapture(ms30)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kernel = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        
        low = 50
        high = 100
        
        edges = cv2.Canny(blur_gray, low, high)
        
        rho = 1 #distance resolution in pixels of the Hough grid
        theta = np.pi/1000 #angular resolution in radians of the Hough grid
        threshold = 30 #minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50 #minimum number of pixels making up a line
        max_line_gap = 50 #maximum gap in pixels between connectable line segments
        line_image = np.copy(frame) * 0
        
        #run Hough on edge detected image 
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        
        upperbound = 0
        lowerbound = 100000
        
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if abs(y1-y2) <3 :
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if y1<lowerbound:
                            lowerbound = y1
                        elif y1>upperbound:
                            upperbound = y1
                        
            cv2.line(line_image, (int(frame.shape[0]/2), upperbound), (int(frame.shape[0]/2), lowerbound), (0, 0, 255), 2)
            lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
        
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 600, 600)
        cv2.imshow('frame', lines_edges)
        cv2.waitKey(1)
        
        frames.append(f)
        f+=1
        distance_pix = upperbound-lowerbound
        if(distance_pix < 0):
            distances.append(np.nan)
        else:
            distance_mm = distance_pix/26
            distances.append(distance_mm)
    else:
        break

cv2.destroyAllWindows()

frames_adjusted = frames[300:1060]
distances_adjusted = distances[300:1060]
plt.plot(frames_adjusted, distances_adjusted, 'b-')
plt.xlabel('Time (s)')
plt.ylabel('Distance (mm)')
plt.title('Distance vs Time (ms30)')

average = np.mean(distances[600:1000])
print(average)
