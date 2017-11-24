# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:30:39 2017
@author: Daria
"""
#from __future__ import division
import numpy as np
import cv2
import math
from skimage.measure import regionprops
from scipy import stats as stts
from leafCheck import leafCheck
from countHWC import CountHeightWidthCoord

sourceImage = cv2.imread("img/6.jpg", cv2.IMREAD_GRAYSCALE);
checkedImage,cnt,coord = leafCheck(sourceImage)

if type(checkedImage) != str:
    ##### Eccentricity #####
    props = regionprops(checkedImage)
    eccentricity = props[0].eccentricity

    #### Circularity ####
    perimeter = cv2.arcLength(cnt,True) 
    area = cv2.contourArea(cnt) 
    circularity = (4*math.pi*area)/(perimeter**2)
    
    ##### Solidity #####
    solidity = props[0].solidity

    ##### Extent #####
    extent = props[0].extent

    ##### Equivalent Diameter #####
    equivalent_diameter = props[0].equivalent_diameter

    ##### Convex hull #####
    convexhull = props[0].convex_area

    # Центроид
    props = regionprops(checkedImage)
    x1 = props[0].centroid[1]
    y1 = props[0].centroid[0]

    # Координаты точек контура
    _, contours, _ = cv2.findContours(checkedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    N = len(cnt)
    dist = []
    for i in range(N):
        p = cnt[i]
        x2 = p.item(0)
        y2 = p.item(1)
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (.5)
        dist.append(distance)

    start = 0
    point = np.where(np.flip(cnt, 2) == coord)

    for i in range(len(np.unique(point[0]))):
        if (np.flip(cnt[np.unique(point[0])[i]], 1) == coord)[0][0] == True & \
                (np.flip(cnt[np.unique(point[0])[i]], 1) == coord)[0][1] == True:
            start = np.unique(point[0])[i]

    newdist = dist[start:N] + dist[0:start]
    arr = np.asarray(newdist)

    # ratio of the leaf area to min incircle
    a = math.pi * min(arr)**2
    f1 = props[0].area / a
    
    #mean
    f2 = np.mean(arr)
    #variance
    f3 = np.var(arr)
    #median
    f4 = np.median(arr)
    #mode
    f5 = stts.mode(arr)[0][0]
    
    #vertical symmetry
    s = len(arr) // 2
    if len(arr) % 2 == 0:
	    f6 = sum(arr[0:s]) / sum(arr[s:len(arr)])
    else:
	    f6 = sum(arr[0:s]) / sum(arr[(s+1):len(arr)])
        
    #horizontal symmetry
    if len(arr) % 2 == 0:
	    mass = arr
    else:
	    mass = arr[0:s] + arr[(s+1):len(arr)]
        
        # 1/4 of array length
    s2 = len(arr) // 4
    if len(mass[0:s]) % 2 == 0:
    	f7 = sum(mass[s2:(len(arr)-s2)]) / sum(mass[(len(mass)-s2):len(mass)] + mass[0:s2])
    else:
        f7 = sum(mass[(s2+1):(len(mass)-s2-1)]) / sum(mass[(len(mass)-s2):len(mass)] + mass[0:s2])
        
    #minimal distance
    f8 = min(arr)/np.mean(arr)
    #maximal distance
    f9 = max(arr)/np.mean(arr)
    #length ratio
    f10 = len(arr)/max(arr)

    heiP1, heiL1, widP1, widL1, maX1, miN1 = CountHeightWidthCoord(arr)
    arr2 = -arr
    heiP2, heiL2, widP2, widL2, maX2, miN2 = CountHeightWidthCoord(arr2)

    ### Number of Peaks ###
    numPeaks = len(maX1)
   # print 'Peaks ', numPeaks

    ### Number of Vals ###
    numVals = len(miN1)
    #print 'Vals ', numVals

    ### Mean Width of Peaks ###
    meanWidthP = np.mean(widL1)
  #  print 'Mean Width of Peaks ', meanWidthP

    ### Mean Height of Peaks ###
    meanHeightP = np.mean(heiL1)
   # print 'Mean Height of Peaks ', meanHeightP

    ### Min Peak ###
    minPeak = min(heiL1)
    #print 'Min Peak ', minPeak

    ### Mean Width of Vals ###
    meanWidthV = np.mean(widL2)
    #print 'Mean Width of Vals ', meanWidthV

    ### Mean Height of Vals ###
    meanHeightV = np.mean(heiL2)
    #print 'Mean Height of Vals ', meanHeightV

    ### Max Val ###
    maxVal = max(heiL2)
    #print 'Max Val ', maxVal
    
    features = [eccentricity, circularity, solidity, extent, equivalent_diameter, 
                convexhull, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                numPeaks, numVals, meanWidthP, meanHeightP, minPeak, meanWidthV,
                meanHeightV, maxVal]
    print "The features were successfully extracted!"

else:
    print checkedImage
    
cv2.waitKey(0)
cv2.destroyAllWindows()