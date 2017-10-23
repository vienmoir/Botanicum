import numpy as np
import cv2

result=0
while result == 0:
    img = cv2.imread('img/1.jpg')
    
    height, width,_ = img.shape
    
    #Бинаризация
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval3, otsu = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    #Удаление мелких объектов
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(otsu, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = height*width*0.1
    print min_size 
    
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    
    #Заполнение мелких дыр в объекте
    closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
    
    #Перевод изображения в рабочую кодировку
    closing = closing.astype(np.uint8)
    
    #Автоматическое выделение контуров
    v = np.median(closing)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(closing, lower, upper)
    
    cv2.imshow('Image_closed', edged)
    
    #Проверка контуров по размеру
    _, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        result=1
        print "Ошибка. Слишком мелкий объект для анализа, сфотографируйте лист крупнее, пожалуйста."
    
    #Удаление незамкнутых контуров
    edged3 = edged.copy()
    mask = np.zeros((np.array(edged.shape)+2), np.uint8)
    cv2.floodFill(edged, mask, (0,0), (255))
    edged = cv2.erode(edged, np.ones((3,3)))
    edged = cv2.bitwise_not(edged)
    edged = cv2.bitwise_and(edged,edged3)
    
    #Проверка оставшихся на изображении контуров
    _, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        result=4
        print "Ошибка. Сфотографируйте лист на нейтральном фоне, пожалуйста."
        
    if len(contours) == 0:
        result=3
        print "Ошибка. Лист выходит за края изображения, сфотографируйте его полностью, пожалуйста."
        
    if len(contours) == 1:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area < min_size:
            result=5
            print "Ошибка. Сфотографируйте лист на контрастном фоне, пожалуйста."
    
    cv2.drawContours(edged,contours,-1,(255,255,255),1)
    
    cv2.imshow('Image', edged)
    
    print result 

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
