# -*- coding: utf-8 -*-
from leafCheck import leafCheck
from processLeaf import process
from classifyLeaf import classify

sourceImage = "D:/Uni/mas/bot/img/ginkgo.jpg"
checkedImage,cnt,coord = leafCheck(sourceImage)
if type(checkedImage) != str:
    print "alright"
    features = process(checkedImage,cnt,coord)
    print "feats extracted"
    result1, result2, result3 = classify(features)
    print "классифицирован"
    print result1
    if result2 != 0:
        print result2
        if result3 != 0:
            print result3
else:
    print checkedImage
