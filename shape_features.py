# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.measure import regionprops

img = cv2.imread("f1.jpg", 0)
ret,thresh = cv2.threshold(img,127,255,0)

#####eccentricity#####
props = regionprops(thresh)
eccentricity = props[0].eccentricity
#print(props[0].eccentricity)

######solidity#####
solidity = props[0].solidity
#print(props[0].solidity)

#######extent#######
extent = props[0].extent
#print(props[0].extent)

#######Equivalent Diameter######
equivalent_diameter = props[0].equivalent_diameter
#print(props[0].equivalent_diameter)

#convex hull
convexhull = props[0].convex_area
#print(props[0].convex_area)

cv2.waitKey(0)
cv2.destroyAllWindows()