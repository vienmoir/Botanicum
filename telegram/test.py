# -*- coding: utf-8 -*-
from leafCheck import leafCheck
from processLeaf import process
from RUclassifyLeaf import classify

sourceImage = "img/1.jpg"
checkedImage,cnt,coord = leafCheck(sourceImage)
if type(checkedImage) != str:
    print "проверено"
    features = process(checkedImage,cnt,coord)
    print "признаки извлечены"
    result1, result2, result3 = classify(features)
    print "классифицирован"
    print result1
    if result2 != 0:
        print result2
        if result3 != 0:
            print result3
else:
    print checkedImage