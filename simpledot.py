# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:01:15 2023

@author: 3mai1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

frames = []
xcoords = []
ycoords = []

movingdotpath = r"C:\Users\3mai1\Downloads\movingdot.mp4"

f = 0
            
print('Enter plot type (path, x, y): ')
plot_type = input()

#open video file for reading
cap = cv2.VideoCapture(movingdotpath)

#iterate through frames of video
while(cap.isOpened()):
    
    #read next frame from video
    ret, frame = cap.read()
    #check if frame was read successfully
    if ret: 
        
        #define lower and upper threshold for white color in BGR
        lowerwhite = np.array([200, 200, 200], dtype = np.uint8)
        upperwhite = np.array([255,255,255], dtype = np.uint8)
        
        #create binary image by thresholding the frame for the white color
        mask = cv2.inRange(frame, lowerwhite, upperwhite)
        
        #find contours in the binary image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #iterate through contours
        for contour in contours:
            #find centroid of contour
            if cv2.contourArea(contour) >10:
                M = cv2.moments(contour)  #finds moment of the contour (statistical measures that describe pixel intensities within the contour)
                centroid_x = int(M['m10']/M['m00'])
                centroid_y = int(M['m01']/M['m00'])
                #to find the x and y coordinates of the centroid, the sum of the product of x-coordinate and intensity
                # and the sum of the product of y-coordinate and intensity is divided by the zeroth moment
                frames.append(f)
                f+=1
                xcoords.append(centroid_x)
                ycoords.append(centroid_y)
                
            
                cv2.circle(frame, (centroid_x, centroid_y), 10, (0,255,0), -1)
            
            
        #display the frame in a window
        cv2.imshow("frame", frame)
        
        cv2.waitKey(1)
    else: 
        #if frame was not read successfully, exit loop
        break
    
cap.release()
cv2.destroyAllWindows()
'''
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('plots')

ax1.plot(xcoords, ycoords)
ax1.xlabel('x-coordinate')
ax1.ylabel('y-coordinate')
           
ax2.plot(frames, xcoords)
ax2.xlabel('frame #')
ax2.ylabel('x-coordinate')

ax3.plot(frames, ycoords)
ax3.xlabel('frame #')
ax3.ylabel('y-coordinate')
'''

#coordinate plot
if plot_type=='path':
    plt.plot(xcoords, ycoords, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Centroid Path')
elif plot_type=='x':
    plt.plot(frames, xcoords, 'g-')
    plt.xlabel('Frame #')
    plt.ylabel('x-coordinate')
    plt.title('x-coordinate by frame')
elif plot_type=='y':
    plt.plot(frames, ycoords, 'g-')
    plt.xlabel('Frame #')
    plt.ylabel('y-coordinate')
    plt.title('y-coordinate by frame')
else:
    raise Exception("invalid plot type")






            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        