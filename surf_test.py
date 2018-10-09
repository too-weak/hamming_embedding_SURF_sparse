import cv2
import numpy as np
img = cv2.imread('D:\Programs\searchPic\gongsi1000\\11.jpg')
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=19500)
key, des = surf.detectAndCompute(img, None)
img = cv2.drawKeypoints(img, key, img)
cv2.imshow('test', img)
cv2.waitKey(0)