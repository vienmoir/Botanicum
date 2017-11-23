# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:30:31 2017

@author: Eclair
"""
import numpy as np
from peakDetect import peakDetect
from operator import itemgetter


def countLen(x1, y1, x2, y2):
        side = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return side

### Function for Peaks detection and heights&lengths evaluation ###
def CountHeightWidthCoord(arr):
    maX, miN = peakDetect(arr, lookahead=10)

    fullArr = []
    for i in range(len(maX)):
        fullArr.append(maX[i])
    for i in range(len(miN)):
        fullArr.append(miN[i])

    newArr = sorted(fullArr, key=itemgetter(0))

    heiP1 = []
    widP1 = []
    # heiP2 = []
    # widP2 = []
    heiL1 = []
    # heiL2 = []
    widL1 = []
    # widL2 = []
    for i in range(len(newArr)):
        if newArr[i] in maX:
            if i != 0 and i != len(newArr) - 1:
                if newArr[i - 1] in miN and newArr[i + 1] in miN:
                    if newArr[i - 1][1] > newArr[i + 1][1]:
                        heiP1.append([[newArr[i][0], newArr[i - 1][1]], [newArr[i][0], newArr[i][1]]])
                        heiL1.append(countLen(newArr[i][0], newArr[i - 1][1], newArr[i][0], newArr[i][1]))
                        widP1.append([[newArr[i][0], newArr[i + 1][1]], [newArr[i + 1][0], newArr[i + 1][1]]])
                        widL1.append(countLen(newArr[i][0], newArr[i + 1][1], newArr[i + 1][0], newArr[i + 1][1]))

                        # heiP2.append([newArr[i][0], newArr[i + 1][1]])
                        # heiL2.append(countLen(newArr[i][0], newArr[i + 1][1], newArr[i][0], newArr[i][1]))
                        # widP2.append([[newArr[i][0], newArr[i - 1][1]], [newArr[i - 1][0], newArr[i - 1][1]]])
                        # widL2.append(countLen(newArr[i][0], newArr[i - 1][1], newArr[i - 1][0], newArr[i - 1][1]))
                    else:
                        heiP1.append([[newArr[i][0], newArr[i + 1][1]], [newArr[i][0], newArr[i][1]]])
                        heiL1.append(countLen(newArr[i][0], newArr[i + 1][1], newArr[i][0], newArr[i][1]))
                        widP1.append([[newArr[i][0], newArr[i - 1][1]], [newArr[i - 1][0], newArr[i - 1][1]]])
                        widL1.append(countLen(newArr[i][0], newArr[i - 1][1], newArr[i - 1][0], newArr[i - 1][1]))

                        # heiP2.append([newArr[i][0], newArr[i - 1][1]])
                        # heiL2.append(countLen(newArr[i][0], newArr[i - 1][1], newArr[i][0], newArr[i][1]))
                        # widP2.append([[newArr[i][0], newArr[i + 1][1]], [newArr[i + 1][0], newArr[i + 1][1]]])
                        # widL2.append(countLen(newArr[i][0], newArr[i + 1][1], newArr[i + 1][0], newArr[i + 1][1]))
            if i == 0:
                if newArr[i + 1] in miN:
                    heiP1.append([[newArr[i][0], newArr[i + 1][1]], [newArr[i][0], newArr[i][1]]])
                    heiL1.append(countLen(newArr[i][0], newArr[i + 1][1], newArr[i][0], newArr[i][1]))
                    widP1.append([[newArr[i][0], newArr[i + 1][1]], [newArr[i + 1][0], newArr[i + 1][1]]])
                    widL1.append(countLen(newArr[i][0], newArr[i + 1][1], newArr[i + 1][0], newArr[i + 1][1]))
            if i == len(newArr) - 1:
                if newArr[i - 1] in miN:
                    heiP1.append([[newArr[i][0], newArr[i - 1][1]], [newArr[i][0], newArr[i][1]]])
                    heiL1.append(countLen(newArr[i][0], newArr[i - 1][1], newArr[i][0], newArr[i][1]))
                    widP1.append([[newArr[i][0], newArr[i - 1][1]], [newArr[i - 1][0], newArr[i - 1][1]]])
                    widL1.append(countLen(newArr[i][0], newArr[i - 1][1], newArr[i - 1][0], newArr[i - 1][1]))
    return heiP1, heiL1, widP1, widL1, maX, miN