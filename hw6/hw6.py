import numpy as np
import cv2
import glob
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from numpy import array


# import modules used here -- sys is a very standard one
import sys



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def convertPoints(pts):

	print "Hello"

	shape = np.shape(pts)
	print shape(1)
	print shape(2)

def dividePoints(pts):

	image_array = []
	
	for i in range(0,len(pts[0])):
			
			divide_by = pts[3,i]
			one = pts[0,i]/divide_by
			two = pts[1,i]/divide_by
			three = pts[2,i]/divide_by
			image_array.append([one, two, three])

	
	np_image_array = np.asarray(image_array)
	
	return np_image_array



def problem3(K, F, one_pts, two_pts):

	print "Hello"
	K = np.matrix(K)
	E = np.transpose(K) * F * K




	U, S, V = np.linalg.svd(E)
	V = np.transpose(V)


	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

	

	S1 = -1*U * Z * np.transpose(U)
	S2 = U * Z * np.transpose(U)
	R1 = U * np.transpose(W) * np.transpose(V)
	R2 = U * W * np.transpose(V)




	temp = np.matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

	M_left = K * temp 



	S = S1
	R = R1

	tlr = np.array([S[2,1], S[0,2], -1*S[0,1]])
	
	T = np.c_[R,tlr]
	

	K = np.matrix(K)
	T = np.matrix(T)
	M_right = K * T



	
	#convertPoints(one_pts)
	import pdb; pdb.set_trace()
	X = cv2.triangulatePoints(M_left, M_right, np.transpose(one_pts), np.transpose(two_pts))

	pts = dividePoints(X)
	pts = np.transpose(pts)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(pts[0], pts[1], pts[2])
	plt.show()

	x = 5


def problem2(one_pts, two_pts):

	
	F = cv2.findFundamentalMat(one_pts, two_pts)
	
	F = F[0]
	return F

def problem1():
	print "Hello world"

	img1 = cv2.imread('imageA.jpg',0)          # queryImage
	img2 = cv2.imread('imageB.jpg',0) 			# trainImage


	img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5) 
	img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
	#Intiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()



	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	


	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)

	

	good = []
	for m,n in matches:
		if m.distance < 0.6*n.distance:
			good.append([m])

	'''
	good2 = []
	for m,n in matches:
		if m.distance < 0.6*n.distance:
			good2.append(m)
	'''
	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,  flags=2)

	plt.waitforbuttonpress(0) # this will wait for indefinite time
	plt.close()
	plt.imshow(img3),plt.show()



# Initialize lists
	list_kp1 = []
	list_kp2 = []

	# For each match...
	for mat in matches:

	    # Get the matching keypoints for each of the images
	    img1_idx = mat[0].queryIdx
	    img2_idx = mat[0].trainIdx

	    # x - columns
	    # y - rows
	    # Get the coordinates
	    (x1,y1) = kp1[img1_idx].pt
	    (x2,y2) = kp2[img2_idx].pt

	    # Append to each list
	    list_kp1.append((x1, y1))
	    list_kp2.append((x2, y2))

	



	one_pts = np.asarray(list_kp1)
	two_pts = np.asarray(list_kp2)

	return one_pts, two_pts










if __name__ == '__main__':
    #main()
    
    K = np.array([[831.09689147/1.0, 0.0/1.0, 508.4419449/1.0], [0.0/1.0, 824.10968861/1.0, 382.54268033/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]])
   
    print "-------------------------------------------------"
    print "PROBLEM 1"
    print "-------------------------------------------------"
   
    one_pts, two_pts = problem1()
    

    print "-------------------------------------------------"
    print "PROBLEM 2"
    print "-------------------------------------------------"
    print ""
    print ""
    print ""
    F = problem2(one_pts, two_pts)

   

    print "-------------------------------------------------"
    print "PROBLEM 3"
    print "-------------------------------------------------"
    print ""
    print ""
    print ""

    problem3(K,F, one_pts, two_pts)
    
    print "-------------------------------------------------"
    print "PROBLEM 4"
    print "-------------------------------------------------"
    print ""
    print ""
    print ""
   


    print "-------------------------------------------------"
    print "PROBLEM 5"
    print "-------------------------------------------------"
    print ""
    print ""
    print ""
    
    