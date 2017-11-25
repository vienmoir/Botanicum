# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:30:39 2017
@author: Daria
"""
import numpy as np
import cv2
import math
from skimage.measure import regionprops
from scipy import stats as stts
from leafCheck import leafCheck
from countHWC import CountHeightWidthCoord
import glob2
import csv

imNum = 0

##DEFINE TREE TYPE##
treeType = 'all'

head = ['Type','Eccentricity','Circularity','Solidity','Extent','Equivalent_diameter',
        'Convex_hull','CircleRatio','Mean','Variance','Median','Mode','Vertical_symmetry','Horizontal_symmetry',
            'Minimal_distance','Maximal_distance','Length_ratio','Peaks_number',
            'Valleys_number','Average_peak_width','Average_peak_height','Minimal_peak',
            'Average_valley_width','Average_valley_height','Maximal_valley']

propNum = len(head)
images = []
for i in glob2.glob('D:/Uni/mas/bot/img/*.*'):
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    images.append(img)

toWrite = []

if images != []:
    for sourceImage in images:
        #matrixLine = np.zeros(propNum)
        matrixLine = []
        #print matrixLine
        result = 0
        checkedImage,cnt,coord = leafCheck(sourceImage)
        if type(checkedImage) != str:
            ##### Eccentricity #####
            props = regionprops(checkedImage)
            eccentricity = props[0].eccentricity
            matrixLine.append(eccentricity)

            #### Circularity ####
            perimeter = cv2.arcLength(cnt,True) 
            area = cv2.contourArea(cnt) 
            circularity = (4*math.pi*area)/(perimeter**2)
            matrixLine.append(circularity)
        
            ##### Solidity #####
            solidity = props[0].solidity
            matrixLine.append(solidity)
        
            ##### Extent #####
            extent = props[0].extent
            matrixLine.append(extent)
        
            ##### Equivalent Diameter #####
            diameter = props[0].equivalent_diameter
            matrixLine.append(diameter)
        
            ##### Convex hull #####
            convexhull = props[0].convex_area
            matrixLine.append(convexhull)
        
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
            matrixLine.append(f1)
                          
            #mean
            f2 = np.mean(arr)
            matrixLine.append(f2)
            
            #variance
            f3 = np.var(arr)
            matrixLine.append(f2)
            
            #median
            f4 = np.median(arr)
            matrixLine.append(f4)
            
            #mode
            f5 = stts.mode(arr)[0][0]
            matrixLine.append(f5)
        
            #vertical symmetry
            s = len(arr) // 2
            if len(arr) % 2 == 0:
                f6 = sum(arr[0:s]) / sum(arr[s:len(arr)])
            else:
                f6 = sum(arr[0:s]) / sum(arr[(s+1):len(arr)])
            matrixLine.append(f6)
    
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
            matrixLine.append(f7)
            
            #minimal distance
            f8 = min(arr)/np.mean(arr)
            matrixLine.append(f8)
            
            #maximal distance
            f9 = max(arr)/np.mean(arr)
            matrixLine.append(f9)
            
            #length ratio
            f10 = len(arr)/max(arr)
            matrixLine.append(f10)
            
            heiP1, heiL1, widP1, widL1, maX1, miN1 = CountHeightWidthCoord(arr)
            arr2 = -arr
            heiP2, heiL2, widP2, widL2, maX2, miN2 = CountHeightWidthCoord(arr2)

            ### Number of Peaks ###
            numPeaks = len(maX1)
            matrixLine.append(numPeaks)
            
            ### Number of Vals ###
            numVals = len(miN1)
            matrixLine.append(numVals)
            
            ### Mean Width of Peaks ###
            meanWidthP = np.mean(widL1)
            matrixLine.append(meanWidthP)
            
            ### Mean Height of Peaks ###
            meanHeightP = np.mean(heiL1)
            matrixLine.append(meanHeightP)

            ### Min Peak ###
            minPeak = min(heiL1)
            matrixLine.append(minPeak)
            
            ### Mean Width of Vals ###
            meanWidthV = np.mean(widL2)
            matrixLine.append(meanWidthV)

            ### Mean Height of Vals ###
            meanHeightV = np.mean(heiL2)
            matrixLine.append(meanHeightV)
    
            ### Max Val ###
            maxVal = max(heiL2)
            matrixLine.append(maxVal)
       
            toWrite.append(matrixLine)
            print "The features were successfully extracted!"

        else:
            print checkedImage


imNum = len(toWrite)
imgMatrix = [[0 for u in range(propNum)] for v in range(imNum+1)]
imgMatrix[0] = head
for i in range(1,imNum+1):
    imgMatrix[i][0] = treeType

for i in range(imNum):
    imgMatrix[i+1][1:propNum] = toWrite[i]

with open(treeType + '.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(imgMatrix)

cv2.waitKey(0)
cv2.destroyAllWindows()