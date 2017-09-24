#!/usr/bin/env python
import numpy as np
import cv2
from mpl_toolkits.mplot3d import axes3d
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



def problemFour():

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot_wireframe(
		[0,4,4,0,0,0,4,4,4,4,4,4,0,0,0,2,0,2,4,2,4,2],
		[0,0,4,4,0,0,0,0,0,4,4,4,4,4,0,2,4,2,4,2,0,2],
		[0,0,0,0,0,4,4,0,4,4,0,4,4,0,4,8,4,8,4,8,4,8], rstride = 10, cstride = 10)

	



	house = np.array([[0,0,0], [4,0,0], [4,4,0], [0,4,0], [0,0,4], [4,0,4], [4,4,4], [0,4,4], [2,2,8]])
	extra = np.array([1, 1, 1, 1, 1, 1, 1, 1,1])
	extra = np.matrix(extra)


	house = np.matrix(house)
	house = house.transpose()
	house = np.vstack([house,extra])



	combined = np.array([[-0.707, -0.707, 0, 3], [0.707, -0.707, 0, 0.5], [0, 0, 1, 3]])
	combined = np.matrix(combined)	

	K = np.array([[100, 0, 200], [-0, 100, 200], [0, 0, 1]])
	K = np.matrix(K)

	pts = K * combined * house;
	pts = pts.transpose()

	image_array = []

	for i in range(0,9):
		#import pdb; pdb.set_trace()
		divide_by = pts.item((i,2))
		one = pts.item((i,0))/divide_by
		two = pts.item((i,1))/divide_by
		image_array.append([one, two])


	
	
	
	np_image_array = np.asarray(image_array)

	pt1 = (int (np_image_array[0,0]), int(np_image_array[0,1]))
	pt2 = (int (np_image_array[1,0]), int(np_image_array[1,1]))
	pt3 = (int (np_image_array[2,0]), int(np_image_array[2,1]))
	pt4 = (int (np_image_array[3,0]), int(np_image_array[3,1]))
	pt5 = (int (np_image_array[4,0]), int(np_image_array[4,1]))
	pt6 = (int (np_image_array[5,0]), int(np_image_array[5,1]))
	pt7 = (int (np_image_array[6,0]), int(np_image_array[6,1]))
	pt8 = (int (np_image_array[7,0]), int(np_image_array[7,1]))
	pt9 = (int (np_image_array[8,0]), int(np_image_array[8,1]))


	image_house = np.zeros((512, 512, 3), np.uint8)
	cv2.line(image_house, pt1, pt2, np.array((0,0,255)), 1)
	cv2.line(image_house, pt2, pt3, np.array((0,0,255)), 1)
	cv2.line(image_house, pt3, pt4, np.array((0,0,255)), 1)
	cv2.line(image_house, pt4, pt1, np.array((0,0,255)), 1)
	cv2.line(image_house, pt1, pt5, np.array((0,0,255)), 1)
	cv2.line(image_house, pt5, pt6, np.array((0,0,255)), 1)
	cv2.line(image_house, pt6, pt7, np.array((0,0,255)), 1)
	cv2.line(image_house, pt7, pt8, np.array((0,0,255)), 1)
	cv2.line(image_house, pt8, pt5, np.array((0,0,255)), 1)

	cv2.line(image_house, pt2, pt6, np.array((0,0,255)), 1)
	cv2.line(image_house, pt3, pt7, np.array((0,0,255)), 1)
	cv2.line(image_house, pt4, pt8, np.array((0,0,255)), 1)

	cv2.line(image_house, pt5, pt9, np.array((0,0,255)), 1)
	cv2.line(image_house, pt8, pt9, np.array((0,0,255)), 1)
	cv2.line(image_house, pt7, pt9, np.array((0,0,255)), 1)
	cv2.line(image_house, pt6, pt9, np.array((0,0,255)), 1)


	while True:
		cv2.imshow("houseimage", image_house)
		key = cv2.waitKey(1) & 0xFF
		plt.show()

		# if the 'c' key is pressed, break from the loop
		if key == ord("q"):
			break

# Close the window will exit the program
	cv2.destroyAllWindows()
	

def problemFourTwo():

	R = np.array([[-0.707, -0.707, 0], [0.707, -0.707, 0], [0, 0, 1]])
	t = np.array([3, 0.5, 3])

	combined = np.array([[-0.707, -0.707, 0, 3], [0.707, -0.707, 0, 0.5], [0, 0, 1, 3]])
	combined = np.matrix(combined)	

	K = np.array([[100, 0, 200], [-0, 100, 200], [0, 0, 1]])
	K = np.matrix(K)

	house = np.array([[100, 300, 0], [300, 300, 0], [300, 500, 0], [100 , 500, 0], [200,150, 0],
		[350,250,0], [400, 250, 0], [375, 100, 0]])

	extra = np.array([1, 1, 1, 1, 1, 1, 1, 1])
	extra = np.matrix(extra)


	house = np.matrix(house)
	house = house.transpose()
	house = np.vstack([house,extra])
	
	
	pts = K * combined * house;
	pts = pts.transpose()
	
	image_array = []

	for i in range(0,8):
		#import pdb; pdb.set_trace()
		divide_by = pts.item((i,2))
		one = pts.item((i,0))/divide_by
		two = pts.item((i,1))/divide_by
		image_array.append([one, two])


	
	print image_array
	
	np_image_array = np.asarray(image_array)

	pt1 = [np_image_array[0,0], np_image_array[0,1]]
	pt2 = [np_image_array[1,0], np_image_array[1,1]]
	pt3 = [np_image_array[2,0], np_image_array[2,1]]
	pt4 = [np_image_array[3,0], np_image_array[3,1]]
	pt5 = [np_image_array[4,0], np_image_array[4,1]]
	pt6 = [np_image_array[5,0], np_image_array[5,1]]
	pt7 = [np_image_array[6,0], np_image_array[6,1]]
	pt8 = [np_image_array[7,0], np_image_array[7,1]]


	plt.plot(pt1, pt2)
	plt.plot(pt1, pt5)
	plt.plot(pt1, pt4)
	plt.plot(pt1, pt6)
	plt.plot(pt2, pt3)
	plt.plot(pt2, pt7)
	plt.plot(pt2, pt5)
	plt.plot(pt3, pt4)
	plt.plot(pt5, pt8)
	plt.plot(pt8, pt7)
	plt.plot(pt8, pt6)
	plt.plot(pt6, pt7)


	plt.show()



	#for i in range (0,9)
	
	



def problemThree():
	
	testimage = cv2.imread('testimg.jpg',1)
	cv2.imshow('testimage',testimage)
	cv2.waitKey(0)

	tiShape = testimage.shape
	xvals = [0,tiShape[1],tiShape[1],0]
	yvals = [0,0, tiShape[0], tiShape[0]]
	

	A = createA_mat(xvals, yvals, xcoor,ycoor)
	H = createH_mat(A)
	
	image = cv2.imread('ts.jpg', cv2.IMREAD_COLOR)

	#Black polygon
	pts = np.array([[xcoor[0],ycoor[0]],[xcoor[1],ycoor[1]],[xcoor[2],ycoor[2]],[xcoor[3],ycoor[3]]], np.int32)
	pts = pts.reshape((-1,1,2))
	cv2.fillPoly(image, [pts], (0,0,0))
	    	
	
	timesShape = image.shape
	dst = cv2.warpPerspective(testimage,H,(timesShape[1],timesShape[0]))

	
	result = dst + image

	while True:
		cv2.imshow('result',result)
		key = cv2.waitKey(1) & 0xFF

		# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break

 	cv2.destroyAllWindows()
	
	
	



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
    problemThree()
    problemFour()




# Select End Points of foreshortened window or billboard

# Set the corresponding point in the frontal view as 

# Estimate the homography 

# Warp the image
# cv.Remap(src, dst, mapx, mapy, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0)) 

#Crop the image




