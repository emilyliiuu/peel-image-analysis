# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:45:09 2023

@author: 3mai1
"""



import cv2
import numpy as np
from scipy.optimize import curve_fit

path = r"C:\Users\3mai1\Downloads\stillframe.png"
xvals = []
yvals = []
distances = []
frames = []
f = 0

frame = cv2.imread(path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# kernel = 5
# blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

low = 50
high = 110

edges = cv2.Canny(gray, low, high)

rho = 1 #distance resolution in pixels of the Hough grid
theta = np.pi/1000 #angular resolution in radians of the Hough grid
threshold = 5 #minimum number of votes (intersections in Hough grid cell)
min_line_length = 20 #minimum number of pixels making up a line
max_line_gap =20 #maximum gap in pixels between connectable line segments
line_image = np.copy(frame) * 0

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

upperboundx = 0
lowerboundx = 100000
upperboundy = 0
lowerboundy = 100000

if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1-x2)>50:
                xvals.append(x1)
                xvals.append(x2)
                yvals.append(y1)
                yvals.append(y2)
                
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if y2>upperboundy:
                    upperboundx = x1
                    upperboundy = y2
                if y1<lowerboundy:
                    lowerboundx = x2
                    lowerboundy = y1
    line_image = cv2.addWeighted(frame, 1, line_image, 1, 0)
    
    #diagonal line
    cv2.line(line_image, (lowerboundx, lowerboundy), (upperboundx, upperboundy), (255, 255, 255), 2)
    lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
    x1, y1 = lowerboundx, lowerboundy
    x2, y2 = upperboundx, upperboundy
    
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    a = int((y2-y1)*-1)
    b = int((x1-x2)*-1)
    c = int(x2*y1-x1*y2)
    cfull = np.full(len(yvals), c)
    print(a, b, c)
    '''
    #vertical line
    cv2.line(line_image, (lowerboundx, lowerboundy), (lowerboundx, upperboundy), (0, 0, 255), 2)
    lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
    #horizontal line
    cv2.line(line_image, (lowerboundx, upperboundy), (upperboundx, upperboundy), (255, 0, 255), 2)
    lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
    '''
    '''
    print("Data types:")
    print("a:", type(a))
    print("b:", type(b))
    print("xvals:", type(xvals))
    print("yvals:", type(yvals))
    
    print("Shapes:")
    print("xvals:", xvals.shape)
    print("yvals:", yvals.shape)
    print(len(a*xvals), len(b*yvals))
    print(a*xvals)
    print(b*yvals)
    print(len(np.abs(a*xvals+b*yvals+cfull)))
    '''
    
    
    distances = np.abs(a*xvals+b*yvals+cfull)/np.sqrt(a**2+b**2)
    mean_distance = np.mean(distances)
    std_deviation = np.std(distances)
    
    std_threshold = 0.6
    filtered_xvals = []
    filtered_yvals = []
    distances_list = []
    
    for x, y, distance in zip(xvals, yvals, distances):
        if np.abs(distance - mean_distance) <= std_threshold * std_deviation:
            filtered_xvals.append(x)
            filtered_yvals.append(y)
            x = int(x)
            y = int(y)
            cv2.circle(line_image, (x, y), 2, (0, 255, 0), 2)
            distances_list.append(distance)
            

    coeffs = np.polyfit(filtered_xvals, filtered_yvals, 2)
    
    curve_x = np.linspace(min(xvals), max(xvals), num = line_image.shape[1])
    curve_y = np.polyval(coeffs, curve_x)
    
    mask = np.zeros_like(line_image, dtype = np.uint8)
    points = np.column_stack((curve_x, curve_y)).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(mask, [points], isClosed = False, color=(0, 0, 255), thickness = 2)
    
    image = cv2.addWeighted(line_image, 1, mask, 1, 0)
        
    '''
    #scipy approach (quadratic)
    coefficients, _ = curve_fit(parabolic_curve, xvals, yvals)
    curve_x = np.linspace(min(xvals), max(xvals), num = lines_edges.shape[1])
    curve_y = parabolic_curve(curve_x, *coefficients)
    
    curve_y = curve_y * (lines_edges.shape[0]/max(curve_y))
    mask = np.zeros_like(lines_edges, dtype= np.uint8)
    points = np.column_stack((curve_x, curve_y)).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(mask, [points], isClosed=False, color = (255, 255, 255), thickness = 2)
    image = cv2.addWeighted(lines_edges, 1, mask, 1, 0)
    '''
    '''
    #scipy approach (exponential)
    initial_guess = [1,1,1]
    parameter_bounts = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
    fit_parameters, _ = curve_fit(exponential_curve, xvals, yvals, p0=initial_guess)
    curve_x = np.linspace(min(xvals), max(xvals), num=line_image.shape[1])
    curve_y = exponential_curve(curve_x, *fit_parameters)
    #curve_y[np.logical_or(np.isnan(curve_y), np.isinf(curve_y))] = 0
    curve_y = curve_y*(line_image.shape[0]/max(curve_y))
    mask = np.zeros_like(line_image, dtype = np.uint8)
    points = np.column_stack((curve_x, curve_y)).reshape((-1,1,2)).astype(np.int32)
    cv2.polylines(mask, [points], isClosed=False, color = (255,255,255), thickness = 2)
    image = cv2.addWeighted(line_image, 1, mask, 0.8, 0)
    '''
# triangle= findtriangle(edges)
# if triangle is not None:
#     cv2.drawContours(edges, [triangle], 0, (255, 0, 0), 5)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 600, 600)
cv2.imshow('frame', image)
cv2.waitKey(0)


cv2.destroyAllWindows()

height = abs(upperboundy-lowerboundy)/100
halfbase = abs(upperboundx-lowerboundx)/100
totalarea = height*halfbase
print("height = " + str(height) + "mm")
print("halfbase = "  + str(halfbase) + "mm")
print("area (assuming symmetric) = " +str(totalarea) + " sq mm")

