# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:47:19 2017

@author: Eclair
"""
import numpy as np
import cv2
import math

<<<<<<< HEAD
def leafCheck(sourceImage):
    result = True

=======
def leafCheck(imageName):
    result = True

    # Read image
    img = imageName
    #print imageName
    sourceImage = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

>>>>>>> master
    # Resize if necessary
    TARGET_PIXEL_AREA = 300000.0
    ratio = float(sourceImage.shape[1]) / float(sourceImage.shape[0])
    new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
    new_w = int((new_h * ratio) + 0.5)
    img = cv2.resize(sourceImage, (new_w,new_h))
    height, width = img.shape
    
    # Threshold 
    th, im_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels > than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    # Connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_out, connectivity=8)
<<<<<<< HEAD
    sizes = stats[1:, -1] 
    nb_components = nb_components - 1
    min_size = height*width*0.08
    max_size = height*width*0.8
=======
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = height*width*0.08
>>>>>>> master
    
    # Remove small objects
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255
    
    img = img.astype(np.uint8)
    
<<<<<<< HEAD
    # Contours
    edgedImage = img.copy()
    _, contours, _ = cv2.findContours(edgedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edgedImage,contours,-1,(255,255,255),1)

    if len(contours) == 0:
        result=False
        error = 'The leaf is too small. Take a closer shot!'
        return error,0,0
        
=======
    #Contours
    edgedImage = img.copy()
    _, contours, _ = cv2.findContours(edgedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edgedImage,contours,-1,(255,255,255),1)
    #cv2.imshow('After contouring', preprocessedImage)
    #
    testlen = len(contours)
    print testlen
    if len(contours) == 0:
        result=False
        error = 'The leaf is too small. Take a closer shot!'
        return error

>>>>>>> master
    else:
        #Tophat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50, 50))
        #kernel = np.ones((15,25),np.uint8)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        thresh = cv2.threshold(tophat, 200, 255, cv2.THRESH_BINARY)[1]
        thresh2 = thresh.copy()
        _, contours,_ = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thresh2,contours,-1,(255,255,255),1)
        #cv2.imshow('Tophat', thresh2)

        if len(contours) == 0:
            result=False
            error = 'Error. Use a neutral background, please!'
<<<<<<< HEAD
            return error,0,0
=======
            return error
>>>>>>> master

        else:
            #
            def findDiag(cont):
                #
                rect = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #
                side = np.sqrt((box[0,0]-box[2,0])**2+(box[2,1]-box[0,1])**2)
                return side

            maxSide=0
            #
            for cont in contours:
                side = findDiag(cont)
                if side > maxSide:
                    maxSide = side

            #     
            for cont in contours:
                #
                mask = np.zeros(thresh.copy().shape,np.uint8)
                cv2.drawContours(mask,[cont],0,255,-1)
                pixelpoints = np.transpose(np.nonzero(mask))
                #
                side = findDiag(cont)
                #
                if side == maxSide:
                    forPoint = pixelpoints
                    for i in pixelpoints:
                        img[i[0],i[1]] = 0

            #
            backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

<<<<<<< HEAD
=======

>>>>>>> master
            #
            for i in forPoint:
                if img[i[0]-1,i[1]] == 255:
                    coord = i[0]-1,i[1]
                    continue
                elif img[i[0],i[1]+1] == 255:
                     coord = i[0],i[1]+1
                     continue
                elif img[i[0]+1,i[1]] == 255:
                     coord = i[0]+1,i[1]
                     continue
                elif img[i[0],i[1]-1] == 255:
                     coord = i[0],i[1]-1
                     continue

            #
            backtorgb[coord[0],coord[1]] = [0,0,255]
            height, width, channels = backtorgb.shape
            cv2.line(backtorgb,(0,coord[0]),(height,coord[0]),(0,0,255),1)
            cv2.line(backtorgb,(coord[1],width),(coord[1],0),(0,0,255),1)

            #
            edgedImage = img.copy()
            _, contours, _ = cv2.findContours(edgedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edgedImage,contours,-1,(255,255,255),1)

            if len(contours) > 1:
                    result=False
                    error = "Error. This doesn't look like a leaf at all!"
                    return error
            else:
                if len(contours) == 1:
                    cnt = contours[0]
                    x,y,w,h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    if x <= 1 or y <=1:
<<<<<<< HEAD
                        result=False
                        error = 'Error. Leaf extends beyond the image borders.'
                        return error,0,0
                    elif x+w-5 >= img.shape[1] or y+h-5 >= img.shape[0]:
                        result=False
                        error = 'Error. Leaf extends beyond the image borders.'
                        return error,0,0
                    elif area < min_size:
                        result=False
                        error = 'Error. Not enough contrast.'
                        return error,0,0
                    elif area > max_size:
                        result=False
                        error = 'Error. Please, take a photo in another illumination.'
                        return error,0,0
                        
    if result == True:
        print "Your leaf was checked successfully!"
        return img,cnt,coord
=======
                        #cv2.imshow('Image', img)
                        result=False
                        error = 'Error. Leaf extends beyond the image borders.'
                        return error
                    elif x+w-5 >= img.shape[1] or y+h-5 >= img.shape[0]:
                        #cv2.imshow('Image', img)
                        result=False
                        error = 'Error. Leaf extends beyond the image borders.'
                        return error
                    elif area < min_size:
                        #cv2.imshow('Image', img)
                        result=False
                        error = 'Error. Not enough contrast.'
                        return error
    if result == True:
        # return img
        return 'What a leaf!'
>>>>>>> master
