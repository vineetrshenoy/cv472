import numpy as np
import cv2
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from numpy import array


# import modules used here -- sys is a very standard one
import sys



def test():
	x = np.array([2, 3, 6, 3, 2, 1, 0, -1, -2, -1, 0, 1, 2])
	16 - x


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def dividePoints(pts):

	image_array = []
		
	for i in range(0,9):
		#import pdb; pdb.set_trace()
		divide_by = pts.item((2,i))
		one = pts.item((0,i))/divide_by
		two = pts.item((1,i))/divide_by
		image_array.append([one, two])

		
	np_image_array = np.asarray(image_array)
	#import pdb; pdb.set_trace()
	#one = np.ones(9)
	#np_image_array = np.vstack((np_image_array,one))

	return np_image_array

def drawMyObject(pts, title, number):
		
		image_array = []
		
		for i in range(0,9):
			#import pdb; pdb.set_trace()
			divide_by = pts.item((2,i))
			one = pts.item((0,i))/divide_by
			two = pts.item((1,i))/divide_by
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

		#import pdb; pdb.set_trace()

		image_house = np.zeros((512, 512, 3), np.uint8)
		#cv2.line(image_house, pt1, pt2, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt2, pt3, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt3, pt4, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt4, pt1, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt5, pt6, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt6, pt7, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt7, pt8, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt8, pt5, np.array((0,0,255)), 1)

		#cv2.line(image_house, pt1, pt5, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt2, pt6, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt3, pt7, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt4, pt8, np.array((0,0,255)), 1)

		#cv2.line(image_house, pt5, pt9, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt8, pt9, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt7, pt9, np.array((0,0,255)), 1)
		#cv2.line(image_house, pt6, pt9, np.array((0,0,255)), 1)

		fig = plt.figure(number)
		fig.canvas.set_window_title(title)
		
		onex = (pt1[0], pt2[0])
		oney = (pt1[1], pt2[1])
		plt.plot(onex, oney)

		onex = (pt2[0], pt3[0])
		oney = (pt2[1], pt3[1])
		plt.plot(onex, oney)
		
		onex = (pt3[0], pt4[0])
		oney = (pt3[1], pt4[1])
		plt.plot(onex, oney)

		onex = (pt4[0], pt1[0])
		oney = (pt4[1], pt1[1])
		plt.plot(onex, oney)

		onex = (pt5[0], pt6[0])
		oney = (pt5[1], pt6[1])
		plt.plot(onex, oney)

		onex = (pt6[0], pt7[0])
		oney = (pt6[1], pt7[1])
		plt.plot(onex, oney)

		onex = (pt7[0], pt8[0])
		oney = (pt7[1], pt8[1])
		plt.plot(onex, oney)

		onex = (pt8[0], pt5[0])
		oney = (pt8[1], pt5[1])
		plt.plot(onex, oney)


		onex = (pt1[0], pt5[0])
		oney = (pt1[1], pt5[1])
		plt.plot(onex, oney)


		onex = (pt2[0], pt6[0])
		oney = (pt2[1], pt6[1])
		plt.plot(onex, oney)

		onex = (pt3[0], pt7[0])
		oney = (pt3[1], pt7[1])
		plt.plot(onex, oney)


		onex = (pt4[0], pt8[0])
		oney = (pt4[1], pt8[1])
		plt.plot(onex, oney)


		plt.waitforbuttonpress(0) # this will wait for indefinite time
		plt.close(fig)
		plt.show()
		
		#return np.image_array
		#while True:
		#	cv2.imshow("houseimage", image_house)
		#	key = cv2.waitKey(1) & 0xFF
		#	plt.show()

			# if the 'c' key is pressed, break from the loop
		#	if key == ord("n"):
		#		break

# Close the window will exit the program
		#cv2.destroyAllWindows()

def problem4():
	K = np.array([[-100, 0, 200], [0, -100, 200], [0, 0, 1]])
	K = np.matrix(K)


	MextLeft = np.array([[0.707, 0.707, 0, -3], [-0.707, 0.707, 0, -0.5], [0, 0, 1, 3]])
	MextLeft = np.matrix(MextLeft)
	MextRight = np.array([[0.866, -0.5, 0, -3], [0.5, 0.866, 0, -0.5], [0, 0, 1, 3]])
	MextRight = np.matrix(MextRight)

	pts = np.array([[2,0,0,1], [3,0,0,1], [3, 1, 0,1], [2, 1, 0,1], [2, 0, 1,1], [3, 0, 1,1], [3, 1, 1,1], [2, 1, 1,1], [2.5, 0.5, 2,1]])
	pts = np.matrix(pts)
	pts = np.transpose(pts)
	NN = 9

	lt = K * MextLeft * pts

	rt = K * MextRight * pts
	drawMyObject(lt, 'Left Camera', 0)
	drawMyObject(rt, 'Right Camera', 0)


	lt = dividePoints(lt)
	rt = dividePoints(rt)
	K_inv = np.linalg.inv(K)

	
	X = cv2.triangulatePoints(K*MextLeft, K*MextRight, np.transpose(lt), np.transpose(rt) )
	
	image_array = []

	for i in range(0,9):
			
			divide_by = X[3,i]
			one = X[0,i]/divide_by
			two = X[1,i]/divide_by
			three = X[2,i]/divide_by
			image_array.append([one, two, three])

	
	image_array = np.transpose(image_array)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	#test = image_array[0,:]
	x_val = image_array[0,:]
	y_val = image_array[1,:]
	z_val = image_array[2,:]

	
	#import pdb; pdb.set_trace()
	#ax.plot_wireframe(testx, testy, testz)
	pt1 = np.array([x_val[0], y_val[0], z_val[0]])
	pt2 = np.array([x_val[1], y_val[1], z_val[1]])
	pt3 = np.array([x_val[2], y_val[2], z_val[2]])
	pt4 = np.array([x_val[3], y_val[3], z_val[3]])
	pt5 = np.array([x_val[4], y_val[4], z_val[4]])
	pt6 = np.array([x_val[5], y_val[5], z_val[5]])
	pt7 = np.array([x_val[6], y_val[6], z_val[6]])
	pt8 = np.array([x_val[7], y_val[7], z_val[7]])
	pt9 = np.array([x_val[8], y_val[8], z_val[8]])


	onex = (pt1[0],pt2[0])
	oney = (pt1[1],pt2[1])
	onez = [pt1[2],pt2[2]]
	ax.plot_wireframe(onex, oney, onez)


	onex = (pt3[0],pt2[0])
	oney = (pt3[1],pt2[1])
	onez = [pt3[2],pt2[2]]
	ax.plot_wireframe(onex, oney, onez)

	onex = (pt3[0],pt4[0])
	oney = (pt3[1],pt4[1])
	onez = [pt3[2],pt4[2]]
	ax.plot_wireframe(onex, oney, onez)


	onex = (pt1[0],pt4[0])
	oney = (pt1[1],pt4[1])
	onez = [pt1[2],pt4[2]]
	ax.plot_wireframe(onex, oney, onez)


	onex = (pt1[0],pt5[0])
	oney = (pt1[1],pt5[1])
	onez = [pt1[2],pt5[2]]
	ax.plot_wireframe(onex, oney, onez)

	onex = (pt6[0],pt5[0])
	oney = (pt6[1],pt5[1])
	onez = [pt6[2],pt5[2]]
	ax.plot_wireframe(onex, oney, onez)


	onex = (pt6[0],pt2[0])
	oney = (pt6[1],pt2[1])
	onez = [pt6[2],pt2[2]]
	ax.plot_wireframe(onex, oney, onez)


	onex = (pt3[0],pt7[0])
	oney = (pt3[1],pt7[1])
	onez = [pt3[2],pt7[2]]
	ax.plot_wireframe(onex, oney, onez)

	onex = (pt6[0],pt7[0])
	oney = (pt6[1],pt7[1])
	onez = [pt6[2],pt7[2]]
	ax.plot_wireframe(onex, oney, onez)

	onex = (pt8[0],pt7[0])
	oney = (pt8[1],pt7[1])
	onez = [pt8[2],pt7[2]]
	ax.plot_wireframe(onex, oney, onez)

	onex = (pt8[0],pt5[0])
	oney = (pt8[1],pt5[1])
	onez = [pt8[2],pt5[2]]
	ax.plot_wireframe(onex, oney, onez)

	onex = (pt8[0],pt4[0])
	oney = (pt8[1],pt4[1])
	onez = [pt8[2],pt4[2]]
	ax.plot_wireframe(onex, oney, onez)



	plt.waitforbuttonpress(0) # this will wait for indefinite time
	plt.close(fig)
	plt.show()


def problem3():
	K = np.array([[-100, 0, 200], [0, -100, 200], [0, 0, 1]])
	K = np.matrix(K)


	MextLeft = np.array([[0.707, 0.707, 0, -3], [-0.707, 0.707, 0, -0.5], [0, 0, 1, 3]])
	MextLeft = np.matrix(MextLeft)
	MextRight = np.array([[0.866, -0.5, 0, -3], [0.5, 0.866, 0, -0.5], [0, 0, 1, 3]])
	MextRight = np.matrix(MextRight)

	pts = np.array([[2,0,0,1], [3,0,0,1], [3, 1, 0,1], [2, 1, 0,1], [2, 0, 1,1], [3, 0, 1,1], [3, 1, 1,1], [2, 1, 1,1], [2.5, 0.5, 2,1]])
	pts = np.matrix(pts)
	pts = np.transpose(pts)
	NN = 9

	

	lt = K * MextLeft * pts

	rt = K * MextRight * pts
	drawMyObject(lt, 'Left Camera', 0)
	drawMyObject(rt, 'Right Camera', 0)




def problem2(F_mat):

	K = np.array([[-100, 0, 200], [0, -100, 200], [0, 0, 1]])
	K = np.matrix(K)


	E = np.transpose(K) * F_mat * K

	print E






def problem1():
	print "Hello world"	

	#import pdb; pdb.set_trace()

	K = np.array([[-100, 0, 200], [0, -100, 200], [0, 0, 1]])
	K = np.matrix(K)


	MextLeft = np.array([[0.707, 0.707, 0, -3], [-0.707, 0.707, 0, -0.5], [0, 0, 1, 3]])
	MextLeft = np.matrix(MextLeft)
	MextRight = np.array([[0.866, -0.5, 0, -3], [0.5, 0.866, 0, -0.5], [0, 0, 1, 3]])
	MextRight = np.matrix(MextRight)

	pts = np.array([[2,0,0,1], [3,0,0,1], [3, 1, 0,1], [2, 1, 0,1], [2, 0, 1,1], [3, 0, 1,1], [3, 1, 1,1], [2, 1, 1,1], [2.5, 0.5, 2,1]])
	pts = np.matrix(pts)
	pts = np.transpose(pts)
	NN = 9

	#ones = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
	#pts = np.vstack([pts, pts])
	#pts = np.matrix(pts)

	pix  = np.zeros((NN,3))
	pix = np.matrix(pix)



	lt = K * MextLeft * pts

	rt = K * MextRight * pts
	#drawMyObject(lt, 'Left Camera', 0)
	#drawMyObject(rt, 'Right Camera', 0)
	

	x = dividePoints(lt)
	xp = dividePoints(rt)


	#import pdb; pdb.set_trace()
	#print x
	#print xp
	mat = np.array([x[0,0]*xp[0,0] , x[0,0] * xp[0,1], x[0,0], x[0,1]*xp[0,0], x[0,1]*xp[0,1],
	x[0,1], xp[0,0], xp[0,1], 1])

	#row2 = np.array([1,2,3,4,5,6,7,8,9])
	#row =  np.vstack((row, row2))

	for i in range(1,9):
		row = np.array([x[i,0]*xp[i,0] , x[i,0] * xp[i,1], x[i,0], x[i,1]*xp[i,0], x[i,1]*xp[i,1],
	x[i,1], xp[i,0], xp[i,1], 1])
		mat = np.vstack((mat,row))

	#print mat

	U, S, V = np.linalg.svd(np.matrix(mat))
	lastCol = V[8,:]
	#x = np.transpose(x)
	#xp = np.transpose(xp)
	#import pdb; pdb.set_trace()
	F = cv2.findFundamentalMat(x, xp)

	#print F

	lastCol = np.reshape(lastCol, (3,3))
	#print lastCol


	return lastCol


if __name__ == '__main__':
    #main()
    #F = problem1()
    #problem2(F)
    #problem3()
    problem4()
    