#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import array


# import modules used here -- sys is a very standard one
import sys
i = 0;
xcoor = [];
ycoor = [];
#A = np.empty([8,9])
newX = [100,800, 800, 100]
newY = [100, 100, 500, 500]

refPt = []
testimage = np.zeros((512, 512, 3), np.uint8)
image = np.zeros((512, 512, 3), np.uint8)
windowName = 'HW Window';
lx = -1
ly = -1
def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, image,lx,ly, xcoor, ycoor, A, newX, newY, i, newRowOne, newRowTwo
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates 
	# performed
	matrix = []
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		print  (x,y)
		lx = x
		ly = y
		xcoor.append(lx)
		ycoor.append(ly)
		
		

def createA_mat(xval, yval, newX, newY):
	A = [];
	for i in range(0,4):
		A.append([-1 *xval[i] , -1 * yval[i], -1, 0, 0, 0, newX[i] * xval[i], newX[i] * yval[i], newX[i]])
		A.append([0, 0, 0, -1 *xval[i] , -1 * yval[i], -1, newY[i] * xval[i], newY[i] * yval[i], newY[i]])

	return A

def createH_mat(A_mat):
	U, s, V = np.linalg.svd(A_mat, full_matrices=True, compute_uv=True)
	shape = np.shape(V)
	V = np.transpose(V)
	#import pdb; pdb.set_trace()
	last_col = len(V[0]) - 1
	solution = V[:,[last_col]]
	solution = np.reshape(solution, (3,3))

	#solution = np.matrix(solution)
	#A_mat = np.matrix(A_mat)
	#zer = A_mat * solution

	#print zer
	 
	return solution
	#hmat =  np.transpose(V) * np.linalg.pinv(s) 
	# * np.transpose(U
	#print hmat



# Gather our code in a main() function
def main():
	# Read Image
	image = cv2.imread('ts.jpg',1);
	# image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)
 	arr = np.array(image)
	x = np.size(arr)
	print arr
# keep looping until the 'q' key is pressed

	while True:
	# display the image and wait for a keypress
	        image = cv2.circle(image,(lx,ly), 10, (0,255,255), -1);
		cv2.imshow(windowName, image)
		key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break
 

	# Close the window will exit the program
	cv2.destroyAllWindows()

# Standard boilerplate to call the main() function to begin
# the program.


def problemThree():
	testimage = cv2.imread('ts.jpg',1);
	# image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback("testimage", click_and_keep)






	A = createA_mat(newX, newY, xcoor, ycoor)
	H = createH_mat(A)

	

	testimage = cv2.imread('testimage.jpg',1);
	#image = cv2.warpPerspective(testimage,H,(700,700))
	#plt.subplot(121),plt.imshow(testimage),plt.title('Input')
	#plt.subplot(122),plt.imshow(image),plt.title('Output')
	#plt.show()



def problemTwo():
	A = createA_mat(xcoor,ycoor,newX, newY)
	H = createH_mat(A)

	image = cv2.imread('ts.jpg',1);
	
	dst = cv2.warpPerspective(image,H,(700,700))


	plt.subplot(121),plt.imshow(image),plt.title('Input')
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	plt.show()


	while True:

		if key == ord("c"):
			break






if __name__ == '__main__':
    main()
    problemTwo()




# Select End Points of foreshortened window or billboard

# Set the corresponding point in the frontal view as 

# Estimate the homography 

# Warp the image
# cv.Remap(src, dst, mapx, mapy, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0)) 

#Crop the image




