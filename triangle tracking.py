# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:16:44 2023

@author: 3mai1
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

frames = []
areas = []

videopath = r"C:\Users\3mai1\Downloads\EF10Side-06152023124023-0000_106kPa.mp4"

'''
def findtriangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #filter contours based on area and number of vertices
    valid_contours = []
    for contour in contours:
        approx=cv2.approxPolyDP(contour, 0.03*cv2.arcLength(contour, True), True)
        if len(approx) == 3 and cv2.contourArea(approx)>100:
            contour_color = image[approx[0][0][1], approx[0][0][0]]
            if np.all(contour_color == [255, 255, 255]):
                valid_contours.append(approx)
    
    #sort thhe valid contours by area in descending order
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse = True)
    
    #get the largest triangle contour
    if valid_contours:
        largest = valid_contours[0]
        return largest
    else:
        return None
'''

def findtriangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04*cv2.arcLength(contour, True), True)
        if len(approx) == 3 and cv2.contourArea(approx)>100:
            valid_contours.append(approx)
            
    valid_contours = sorted(valid_contours, key = cv2.contourArea, reverse = True)
    
    if valid_contours:
        largest = valid_contours[0]
        return largest
    else:
        return None

'''
def findtriangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    #apply harris corner detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    #dilate the corners to enhance detection
    corners= cv2.dilate(corners, None)
    
    #threshold the corner response
    threshold = 0.01*corners.max()
    corner_mask = corners > threshold
    
    #find centroids of the corners
    contours, _ = cv2.findContours(corner_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] !=0:
            centroid_x = int(moments['m10']/moments['m00'])
            centroid_y = int(moments['m01']/moments['m00'])
            centroids.append((centroid_x, centroid_y))
            
    #find triangles using the ceentroids
    valid_triangles = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            for k in range(j+1, len(centroids)):
                valid_triangles.append([centroids[i], centroids[j], centroids[k]])
                
    largest_triangle = max(valid_triangles, key = cv2.contourArea)
    return largest_triangle
'''

cap = cv2.VideoCapture(videopath)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        #up the contrast of each frame
        '''
        #this one increases contrast way too much
        frame_eq = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(frame_eq)
        channels = list(channels)
        channels[0] = cv2.equalizeHist(channels[0])
        channels = tuple(channels)
        frame_eq = cv2.merge(channels)
        frame_eq = cv2.cvtColor(frame_eq, cv2.COLOR_YCrCb2BGR)
        '''
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #CLAHE = Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        
        frame_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 600, 400)
        
        triangle = findtriangle(frame_eq)
        if triangle is not None:
            cv2.drawContours(frame_eq, [triangle], 0, (0, 255, 0), 5)
        cv2.imshow('frame', frame_eq)
        cv2.waitKey(1)
        print('here')
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
    