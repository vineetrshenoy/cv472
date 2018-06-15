import sys
import cv2
import numpy as np
import cv2
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from numpy import array



def main():

	image = cv2.imread('ts.jpg', 1)
	
	imageA = cv2.imread('imageA.jpg', 1)
	imageA = cv2.resize(imageA, (0,0), fx=0.25, fy=.25)
	cv2.imshow("newWindown",imageA)
	key = cv2.waitKey(0) 


	imageB = cv2.imread('imageA(1).jpg', 1)
	imageB = cv2.resize(imageB, (0,0), fx=0.25, fy=0.25)
	cv2.imshow("newWindowB",imageB)
	key = cv2.waitKey(0)


	sift = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = sift.detectAndCompute(imageA,None)
	kp2, des2 = sift.detectAndCompute(imageB,None)


	bf = cv2.BFMatcher()

	matches = bf.knnMatch(des1, des2, k=2)


	goodPoints = []
	for m,n in matches:
		if m.distance < 0.5*n.distance:
			goodPoints.append([m])


	result = cv2.drawMatchesKnn(imageA, kp1, imageB, kp2, goodPoints, None, flags =2)

	plt.imshow(result)
	plt.show()







if __name__ == '__main__':

	main()

