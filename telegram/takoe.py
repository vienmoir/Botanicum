# -*- coding: utf-8 -*-
from leafCheck import leafCheck
from processLeaf import process
from classifyLeaf import classify

sourceImage = "D:/Uni/mas/bot/img/redm.jpg"
checkedImage,cnt,coord = leafCheck(sourceImage)
if type(checkedImage) != str:
    print "alright"
    features = process(checkedImage,cnt,coord)
    print "feats extracted"
    result = classify(features)
    print "классифицирован"
    print result
else:
    print checkedImage
