import numpy as np
import cv2
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from numpy import array


# import modules used here -- sys is a very standard one
import sys




def problemOne():
	print "Hello"


def main():
	print "Hello from main()"
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot_wireframe(
			[0,4,4,0,0,0,4,4,4,4,4,4,0,0,0,2,0,2,4,2,4,2],
			[0,0,4,4,0,0,0,0,0,4,4,4,4,4,0,2,4,2,4,2,0,2],
			[0,0,0,0,0,4,4,0,4,4,0,4,4,0,4,8,4,8,4,8,4,8], rstride = 10, cstride = 10)

	#plt.show()

	#House matrix
	house = np.array([[0,0,0,1], [4,0,0,1], [4,4,0,1], [0,4,0,1], [0,0,4,1], [4,0,4,1], [4,4,4,1], [0,4,4,1], [2,2,8,1]])
	house = np.matrix(house)
	house = house.transpose()
	
	## Translation matrix
	camera_rot = np.array([[-0.707, -0.707, 0, 3], [0.707, -0.707, 0, 0.5], [0, 0, 1, 3]])
	camera_rot = np.matrix(camera_rot)

	#intrinsic camera mat
	K = np.array([[100, 0, 200], [-0, 100, 200], [0, 0, 1]])
	K = np.matrix(K)

	M = K * camera_rot

	#rotation matrix for the hosue around the z-axis
	x = np.pi/4

	rotation_mat = np.array([[np.cos(x), -1 * np.sin(x), 0, 0], [np.sin(x), np.cos(x), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	rotation_mat = np.matrix(rotation_mat)
	
	############################################################################



	#rotate the house around z by pi/2
	
	new_house = rotation_mat * house
	

		
	pts = M * new_house

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






if __name__ == '__main__':
    main()
    problemOne()