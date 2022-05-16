##practica 6
import numpy as np
from matplotlib import pyplot as plt
import math as ma
import cv2 #opencv

cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

'''
    #verde
    lower_red = np.array([0,150,50])
    upper_red = np.array([100,255,180])


    #rojo
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    

    #amarillo
    lower_red = np.array([20,120,200])
    upper_red = np.array([100,200,250])
    

    #verde y azul claritos
    lower_red = np.array([10,200,130])
    upper_red = np.array([100,255,250])


    #negro yoscuros
    lower_red = np.array([50,10,3])
    upper_red = np.array([255,100,80])

    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #azul con yuv
    lower_red = np.array([0,150,50])
    upper_red = np.array([100,255,180])


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #rojo con yuv
    lower_red = np.array([20,120,200])
    upper_red = np.array([100,200,250])

'''
