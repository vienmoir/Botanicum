import numpy as np
import cv2

img = cv2.imread('img/4.jpg')
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
######## otsu binarization ########
retval3, otsu = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#cv2.imshow('otsu', otsu)

##### removing small objects ######
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(otsu, connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1
min_size = 30000 

img2 = np.zeros((output.shape))
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255
#cv2.imshow('wll',img2)
#kernel = np.ones((5,5),np.uint8)

###### closing small holes inside the foreground objects or small black points on the object ##########
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
cv2.imshow('closing', closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
