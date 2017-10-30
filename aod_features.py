# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
import statistics
import math

#props = regionprops(thresh)
# example array
arr = [1, 2, 8, -1, 10, -2, 6, 5, 1, 1, 2]

# ratio of the leaf area to something
a = math.pi * min(arr)**2
#f1 =props[0].area / a
#mean
f2 = np.mean(arr)
#variance
f3 = np.var(arr)
#median
f4 = np.median(arr)
#mode
f5 = statistics.mode(arr)
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
f8 = min(arr)/statistics.mean(arr)
#maximal distance
f9 = max(arr)/statistics.mean(arr)
#length ratio
f10 = len(arr)/max(arr)