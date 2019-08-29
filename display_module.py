import numpy as np
import cv2

def display_keypoints(img, kp, path):
	print("Displaying keypoints")
	img2=img.copy()
	cv2.drawKeypoints(img, kp, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite(path+ ".jpg", img2)