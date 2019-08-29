import numpy as np
import cv2

img = cv2.imread('/home/sumedh/Documents/Priyanka/image_classification/Image/0/5_B07CNZCB46.jpg')
img2 = cv2.imread('/home/sumedh/Documents/Priyanka/image_classification/Image/0/708_B07CQXJ6FW.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
print(corners)
print(len(corners))
corners2 = cv2.goodFeaturesToTrack(gray2,25,0.01,10)
corners2 = np.int0(corners2)
#print(corners2)
print(len(corners2))