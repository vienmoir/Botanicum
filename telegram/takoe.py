# -*- coding: utf-8 -*-
#from leafCheck import leafCheck
#from processLeaf import process
#from RUclassifyLeaf import classify
#
#sourceImage = "D:/Uni/mas/bot/img/ginkgo.jpg"
#checkedImage,cnt,coord = leafCheck(sourceImage)
#if type(checkedImage) != str:
#    print "alright"
#    features = process(checkedImage,cnt,coord)
#    print "feats extracted"
#    result1, result2, result3 = classify(features)
#    print "классифицирован"
#    print result1
#    if result2 != 0:
#        print result2
#        if result3 != 0:
#            print result3
#else:
#    print checkedImage
#    
def up(name): return name.decode('utf-8').capitalize()
myfile = open("trees.txt")
msg = myfile.read()
myfile.close()
keyboard = map(up, msg.split('\n'))
test = msg.split('\n')
test[1:21] = test[0:20]
test2 = up(test[2])
L = ("hello", "what", "is", "your", "name")
(L[0].capitalize(),) + L[1:]
print L