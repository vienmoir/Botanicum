import cv2
#import numpy as np
from skimage.measure import regionprops

img = cv2.imread("img/2.jpg", 0)
ret,thresh = cv2.threshold(img,127,255,0)

#centroid
props = regionprops(thresh)
x1 = props[0].centroid[1]
y1 = props[0].centroid[0]
print(x1,y1)

# координаты точек контура
_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
print(len(cnt))

# вектор расстояний dist от центра масс до границ
dist = [] 
for i in range(len(cnt)): 
	p = cnt[i]
	x2 = p.item(0)
	y2 = p.item(1)
	distance = ((x2 - x1)**2 + (y2 - y1)**2)**(.5)
	dist.append(distance)

print type(dist)