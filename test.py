# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:49:00 2023

@author: 3mai1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

frames = []
areas = []

#videopath = r"C:\Users\3mai1\Downloads\stillframe.png"
videopath = r"C:\Users\3mai1\Downloads\stillframe2.png"

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
    corners = cv2.cornerHarris(gray, blockSize = 2, ksize = 3, k = 0.04)
    threshold = 0.01*corners.max()
    corner_points = np.where(corners>threshold)
    corner_points = np.column_stack((corner_points[1], corner_points[0]))
'''
def edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_image = np.zeros_like(image)
    edges_image[edges!=0] = (255, 255, 255)
    return edges_image

def corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k =0.04)
    threshold = 0.0005*corners.max()
    corners_image = np.zeros_like(image)
    corners_image[corners>threshold] = (255, 255, 255)
    return corners_image
    



image = cv2.imread(videopath)

image = image[700:850, 0:1353]

edge = corners(image)
cv2.imshow('frame', edge)


'''
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#CLAHE = Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)

frame_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("frame", 600, 400)

triangle = findtriangle(frame_eq)
if triangle is not None:
    cv2.drawContours(frame_eq, [triangle], 0, (0, 255, 0), 2)
cv2.imshow('frame', frame_eq)
cv2.waitKey(0)

cv2.destroyAllWindows()
'''
triangle = findtriangle(edge)
if triangle is not None:
    cv2.drawContours(edge, [triangle], 0, (0, 255, 0), 2)
cv2.imshow('frame', edge)
cv2.waitKey(0)

cv2.destroyAllWindows()


    