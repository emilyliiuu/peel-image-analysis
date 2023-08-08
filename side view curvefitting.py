# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:51:48 2023

@author: 3mai1
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt

ef10 = r"C:\Users\3mai1\Downloads\EF10Side-06152023124023-0000_106kPa (1).mp4"
ef20 = r"C:\Users\3mai1\Downloads\EF20Side-06152023130307-0000_112kPa.mp4"
ef30 = r"C:\Users\3mai1\Downloads\EF30Side-06152023132504-0000_180kPa.mp4"
ef50 = r"C:\Users\3mai1\Downloads\EF50Side-06152023134549-0000_240kPa.mp4"
efgel = r"C:\Users\3mai1\Downloads\EFGelSide-06152023121714-0000_37kPa.mp4"
ms30 = r"C:\Users\3mai1\Downloads\MS30Side-06152023140706-0000_663kPa.mp4"



areas = []
frames = []

f = 0

cap = cv2.VideoCapture(ef20)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        xvals = []
        yvals = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kernel = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        
        low = 50
        high = 150
        
        edges = cv2.Canny(gray, low, high)
        
        rho = 1 #distance resolution in pixels of the Hough grid
        theta = np.pi/700 #angular resolution in radians of the Hough grid
        threshold = 50 #minimum number of votes (intersections in Hough grid cell)
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
                    if abs(x1-x2)>50 and y1-y2>0:
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
            try:
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
                
                distances = np.abs(a*xvals+b*yvals+cfull)/np.sqrt(a**2+b**2)
                mean_distance = np.mean(distances)
                std_deviation = np.std(distances)
                
                std_threshold = 0.5
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
                #numpy approach
                #coeffs = np.polyfit(xvals, yvals, 2)
                
                curve_x = np.linspace(min(xvals), max(xvals), num = line_image.shape[1])
                curve_y = np.polyval(coeffs, curve_x)
                
                mask = np.zeros_like(line_image, dtype = np.uint8)
                points = np.column_stack((curve_x, curve_y)).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(mask, [points], isClosed = False, color=(0, 0, 255), thickness = 2)
                
                line_image = cv2.addWeighted(line_image, 1, mask, 1, 0)
                
                f+=1
                frames.append(f)
                coeffs = coeffs/100
                upperboundx = upperboundx/100
                lowerboundx = lowerboundx/100
                
                area_upperbound = coeffs[0]*upperboundx + coeffs[1]*upperboundx*upperboundx/2 + coeffs[2]*upperboundx*upperboundx*upperboundx
                area_lowerbound = coeffs[0]*lowerboundx + coeffs[1]*upperboundx*upperboundx/2 + coeffs[2]*lowerboundx*lowerboundx*lowerboundx
                area = area_lowerbound-area_upperbound
                areas.append(area)
            except:
                pass
        
        '''
        #diagonal line
        cv2.line(line_image, (lowerboundx, lowerboundy), (upperboundx, upperboundy), (255, 255, 255), 2)
        lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
        '''
        '''
        #vertical line
        cv2.line(line_image, (lowerboundx, lowerboundy), (lowerboundx, upperboundy), (0, 0, 255), 2)
        image = cv2.addWeighted(image, 1, line_image, 1, 0)
        #horizontal line
        cv2.line(line_image, (lowerboundx, upperboundy), (upperboundx, upperboundy), (255, 0, 255), 2)
        image = cv2.addWeighted(image, 1, line_image, 1, 0)
        '''
        '''
        height = abs(upperboundy-lowerboundy)/100
        halfbase = abs(upperboundx-lowerboundx)/100
        totalarea = height*halfbase
        print("height = " + str(height) + "mm")
        print("halfbase = "  + str(halfbase) + "mm")
        print("area (assuming symmetric) = " +str(totalarea) + " sq mm")
        areas.append(totalarea)
        '''
    else:
        break
        
    # triangle= findtriangle(edges)
    # if triangle is not None:
    #     cv2.drawContours(edges, [triangle], 0, (255, 0, 0), 5)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 600, 600)
    cv2.imshow('frame', line_image)
    cv2.waitKey(1)


cv2.destroyAllWindows()



areas_adjusted = areas[300:700]
frames_adjusted = frames[300:700]
plt.plot(frames_adjusted, areas_adjusted, 'b-')
plt.xlabel('Time (s)')
plt.ylabel('Area (sq mm)')
plt.title('Area vs Time (ef50)')

average = np.nanmean(areas[700:800])
print(average)

