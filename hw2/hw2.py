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
polygon = np.zeros((1000,1000,3), np.uint8)
windowName = 'HW Window';
lx = -1
ly = -1
def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, image, testimage, lx,ly, xcoor, ycoor, A, newX, newY, i, newRowOne, newRowTwo,polygon
 
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
	print "Image shape is: ", image.shape
	# image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)
 	arr = np.array(image)
	x = np.size(arr)
	#print arr
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
	
	testimage = cv2.imread('testimg.jpg',1)

	x = testimage.shape
	print x
	xvals = [0,x[1],x[1],0]
	yvals = [0,0, x[0], x[0]]
	print "xvals are: ", xvals
	print "yvals are: ",yvals

	A = createA_mat(xvals, yvals, xcoor,ycoor)
	H = createH_mat(A)


	#Black polygon
	pts = np.array([[xcoor[0],ycoor[0]],[xcoor[1],ycoor[1]],[xcoor[2],ycoor[2]],[xcoor[3],ycoor[3]]], np.int32)
	#pts = pts.reshape((-1,1,2))
	cv2.polylines(polygon,[pts],True,(0,255,255))

	    	
	cv2.imshow("newwindow", polygon)
	cv2.waitKey()

	#dst = cv2.warpPerspective(testimage,H,(700,700))

	#plt.subplot(121),plt.imshow(testimage),plt.title('Input')
	#plt.subplot(122),plt.imshow(dst),plt.title('Output')
	#plt.show()


	#A = createA_mat(newX, newY, xcoor, ycoor)
	#H = createH_mat(A)

	

	



def problemTwo():
	A = createA_mat(xcoor,ycoor,newX, newY)
	H = createH_mat(A)

	image = cv2.imread('ts.jpg',1);
	
	dst = cv2.warpPerspective(image,H,(1000,1000))


	#plt.subplot(121),plt.imshow(image),plt.title('Input')
	#plt.subplot(122),plt.imshow(dst),plt.title('Output')
	#plt.show()

	while True:
		cv2.imshow("Warp Perspective", dst)
		key = cv2.waitKey(1) & 0xFF

		# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break

# Close the window will exit the program
	cv2.destroyAllWindows()






	





if __name__ == '__main__':
    main()
    problemTwo()




# Select End Points of foreshortened window or billboard

# Set the corresponding point in the frontal view as 

# Estimate the homography 

# Warp the image
# cv.Remap(src, dst, mapx, mapy, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0)) 

#Crop the image




