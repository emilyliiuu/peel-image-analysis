# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:51:27 2023

@author: 3mai1
"""

import cv2
import numpy as np

path = r"C:\Users\3mai1\Downloads\frontview.png"
image= cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = 5
blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

low = 50
high = 70

edges = cv2.Canny(blur_gray, low, high)

rho = 1 #distance resolution in pixels of the Hough grid
theta = np.pi/25 #angular resolution in radians of the Hough grid
threshold = 15 #minimum number of votes (intersections in Hough grid cell)
min_line_length = 50 #minimum number of pixels making up a line
max_line_gap = 20 #maximum gap in pixels between connectable line segments
line_image = np.copy(image) * 0

#run Hough on edge detected image 
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

upperbound = 0
lowerbound = 100000

for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(y1-y2) <4:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if y1<lowerbound:
                lowerbound = y1
            elif y1>upperbound:
                upperbound = y1
            
cv2.line(line_image, (int(image.shape[0]/2), upperbound), (int(image.shape[0]/2), lowerbound), (0, 0, 255), 2)
lines_edges = cv2.addWeighted(image, 1, line_image, 1, 0)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 600, 600)
cv2.imshow('frame', lines_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

distance_pix = upperbound-lowerbound
distance_mm = distance_pix/26
