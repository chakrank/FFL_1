# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:17:27 2018

Enviroment test
To verify which packages are present

@author: chakrank
"""
import cv2
import matplotlib.image as mpimg
import numpy as np

def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3 #5 or 9 these apply the average of a the matrix
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0) #0
    # Define our parameters for Canny and apply
    low_threshold = 30 #40
    high_threshold = 90 #80
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return edges

def mask_image(image):
    edge = gray(image)
    mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))   
    ignore_mask_color = (255,255,255)
    imshape = image.shape
    y_upper = imshape[0]*0.65
    y_bottom = imshape[0] #0.9Bottom is also chosen to avoid hood of the car
    x_left_upper = imshape[1]*0.4
    x_right_upper = imshape[1]*0.6
    vertices = np.array([[(imshape[1]*0.05,y_bottom),(x_left_upper, y_upper), (x_right_upper, y_upper),(imshape[1]*0.95,y_bottom)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edge, mask)
    return masked_edges

def lines_image(image):
    masked_edges = mask_image(image)
    
    rho = 1 #1 distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     #1 minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5 #10 minimum number of pixels making up a line
    max_line_gap = 2    #2 maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
   
    # To see the Hough's marking uncomment the following
    """
    edges = gray(image)
    line_image = np.copy(image)*0 
    for line in lines:
        for x1,y1,x2,y2 in line:
            h_lines = cv2.line(line_image,(x1,y1),(x2,y2),[0,0,255],3)
    
    color_edges = np.dstack((edges, edges, edges))
    lines_edges = cv2.addWeighted(color_edges, 0.8, h_lines, 1, 0)   
    cv2.imshow("h_lines",lines_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

   # Iterate over the output "lines" and draw lines on a blank image
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    slope_left = []
    slope_right = []
    constant_left = []
    constant_right = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 == x2:
                m = np.Infinity
            else:
                m = round(((-1)*(y2-y1)/(x2-x1)), 2)
            if m > 0.43 and m < 1.57:
                x_left.append(x1)
                y_left.append(y1)
                x_left.append(x2)
                y_left.append(y2)
                slope_left.append(m)
                c1 = round(y1 - m*x1,2)
                c2 = round(y2 - m*x2,2)
                constant_left.append(c1)
                constant_left.append(c2)
            elif m < -0.43 and m > -1.57:
                x_right.append(x1)
                y_right.append(y1)
                x_right.append(x2)
                y_right.append(y2)
                slope_right.append(m)
                c1 = round(y1 - m*x1,2)
                c2 = round(y2 - m*x2,2)
                constant_right.append(c1)
                constant_right.append(c2)
            else:
                None
    co_ordinates = [x_left, x_right, y_left, y_right, slope_left,  slope_right, constant_left, constant_right]
    return co_ordinates

def guide_lines(image):
    edges = gray(image)
    line_image = np.copy(image)*0 
    imshape = image.shape
    y_top = np.int32(imshape[0]*0.65)
    y_bottom = imshape[0]
    [x_left, x_right, y_left, y_right, slope_left,  slope_right, constant_left, constant_right] = lines_image(image)

    left_slope = round(np.mean(slope_left),2)
    right_slope = round(np.mean(slope_right), 2)
    
    left_constant = round(np.mean(constant_left),2)
    right_constant = round(np.mean(constant_right), 2)
    
    cl = round(y_bottom - left_slope*275,2)
    cr = round(y_bottom - right_slope*1140,2)
    
    
    # These are used just for presepective   
    #x_left_bottom = 275
    #x_left_top = 520
    #x_right_top = 720 
    #x_right_bottom = 1140
    
    # The follwoing uses maximum conditon
    x_left_top = max(x_left)
    x_left_bottom =  min(x_left)
    x_right_top = min(x_right)
    x_right_bottom = max(x_right)
    
    #The following uses the conditon y = m*x + c and ((y2-y1)/(x2-x1))= m
    """
    x_left_bottom = np.float32(round(((y_bottom - left_constant))/left_slope ,2))
    x_left_top = x_left_bottom - ((y_top-y_bottom)/left_slope)
    x_left_top = np.float32(round(x_left_top))
     Righty
    x_right_bottom = np.float32(round((y_bottom - right_constant)/right_slope,2))
    x_right_top = x_right_bottom - ((y_top-y_bottom)/right_slope) 
    x_right_top = np.float32(round(x_right_top))
    """
    
       
    line_r = cv2.line(line_image, (x_right_bottom, imshape[0]),
                      (x_right_top, y_top), (0, 0, 255), 10)
    #Plottin Left line on the above image
    line_LnR = cv2.line(line_r, (x_left_bottom, imshape[0]),
                        (x_left_top, y_top), (0, 0, 255), 10)

    # Iterate over the output "lines" and draw lines on a blank image
        
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_LnR, 1, 0)
    return lines_edges

file_name = "solidWhiteRight.mp4"
#solidWhiteRight.mp4  solidYellowLeft.mp4  challenge.mp4
cap = cv2.VideoCapture(file_name)
# Width of the frames in the video stream.
frame_width = int(cap.get(3))
# Height of the frames in the video stream
frame_height = int(cap.get(4))
#Frame rate
fps = int(cap.get(5))
#4-character code of codec
fourcc = (cv2.VideoWriter_fourcc('F','M','P','4'))
out = cv2.VideoWriter('solidWhiteRightOut.mp4',fourcc, fps, (frame_width,frame_height))

while True:
    ret, frame = cap.read()
    if ret == True:
        frame_out = guide_lines(frame)
        out.write(frame)
        #print("in\n>", frame,"out\n>", frame_out)
        #Displyaing the resulting frame
        cv2.imshow("frame", frame_out)
        #Press Q to exit
        if cv2.waitKey(1) & 0XFF == ("q"):
            break
    else:
        break
#When everything is done release the video capturing object and out object
cap.release()
out.release()

cv2.destroyAllWindows()

