# -*- coding: utf-8 -*-
from leafCheck import leafCheck
from processLeaf import process
from classifyLeaf import classify

sourceImage = "img/2.jpg"
checkedImage,cnt,coord = leafCheck(sourceImage)
if type(checkedImage) != str:
    print "checked"
    features = process(checkedImage,cnt,coord)
    print "features extracted"
    print features
    result1, result2, result3 = classify(features)
    print "classified"
    print result1
    if result2 != 0:
        print result2
        if result3 != 0:
            print result3
else:
    print checkedImage