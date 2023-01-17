# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:52:21 2022

Converts simulator raw images to lane lines using opencv

@author: Marcus
"""
import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import edge_detection as edge # Handles the detection of lane lines
import matplotlib.pyplot as plt # Used for plotting and error checking
import os
import shutil


def remove_isolated_pixels(image, cluster_size):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    
    new_image = image.copy()
    
    

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] <= cluster_size:
            new_image[labels == label] = 0
        

    return new_image

filename = './224_by_224'

def get_line_markings(frame):
    """
    Isolates lane lines.
   
      :param frame: The camera frame that contains the lanes we want to detect
    :return: Binary (i.e. black and white) image containing the lane lines.
    """
             
    # Convert the video frame from BGR (blue, green, red) 
    # color space to HLS (hue, saturation, lightness).
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
 
    ################### Isolate possible lane line edges ######################
         
    # Perform Sobel edge detection on the L (lightness) channel of 
    # the image to detect sharp discontinuities in the pixel intensities 
    # along the x and y axis of the video frame.             
    # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
    # Relatively light pixels get made white. Dark pixels get made black.
    _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(200, 255))
    sxbinary = edge.blur_gaussian(sxbinary, ksize=3) # Reduce noise
         
    # 1s will be in the cells with the highest Sobel derivative values
    # (i.e. strongest lane line edges)
    sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
 
    ######################## Isolate possible lane lines ######################
   
    # Perform binary thresholding on the S (saturation) channel 
    # of the video frame. A high saturation value means the hue color is pure.
    # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
    # and have high saturation channel values.
    # s_binary is matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the purest hue colors (e.g. >80...play with
    # this value for best results).
    s_channel = hls[:, :, 2] # use only the saturation channel data
    _, s_binary = edge.threshold(s_channel, (100, 255))
     
    # Perform binary thresholding on the R (red) channel of the 
        # original BGR video frame. 
    # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the richest red channel values (e.g. >120).
    # Remember, pure white is bgr(255, 255, 255).
    # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
    _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(180, 255))
 
    # Lane lines should be pure in color and have high red channel values 
    # Bitwise AND operation to reduce noise and black-out any pixels that
    # don't appear to be nice, pure, solid colors (like white or yellow lane 
    # lines.)       
    rs_binary = cv2.bitwise_and(s_binary, r_thresh)
 
    ### Combine the possible lane lines with the possible lane line edges ##### 
    # If you show rs_binary visually, you'll see that it is not that different 
    # from this return value. The edges of lane lines are thin lines of pixels.
    lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(
                              np.uint8))    
    
    #cv2.imshow('test', lane_line_markings)
    
    # Display the window until any key is pressed
    #cv2.waitKey(0) 
       
    # Close all windows
    #cv2.destroyAllWindows() 
    
    kernel = np.array([ [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]],np.uint8)
    
   
    lane_line_markings = cv2.morphologyEx(lane_line_markings, cv2.MORPH_CLOSE, kernel)
    
    
    lane_line_markings[lane_line_markings == 1] = 0
    
    
    lane_line_markings = remove_isolated_pixels(lane_line_markings, cluster_size = 10)
    
    
    return lane_line_markings

def main():
    
    start_path = './224_by_224'
    end_path = './224_224_binary'
    count = 0
    

    for file_name in os.listdir(start_path):
        if not os.path.isdir(end_path + '/' + str(file_name)) and file_name != 'data.csv':
            os.mkdir(end_path + '/' + str(file_name))
        if file_name != 'data.csv':
            for seq in os.listdir(start_path + '/' + file_name):
                current_file = start_path + '/' + file_name + '/' + seq
                saved_path = end_path + '/' + file_name + '/' + seq
                original_frame = cv2.imread(current_file)
                lane_line_markings = get_line_markings(original_frame)
                cv2.imwrite(saved_path, lane_line_markings)
  
  
main()