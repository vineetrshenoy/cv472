import numpy as np
import cv2
import glob
import keras
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from numpy import array
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


# import modules used here -- sys is a very standard one
import sys



def problem3():

	keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

	model = ResNet50(weights='imagenet')

	img_path = 'chair1.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	
	images = glob.glob('*.jpg')

	for fname in images:

		img = image.load_img(fname, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		print "-------------------"
		print "FILE NAME: " 
		print fname
		print "-------------------"
		preds = model.predict(x)
		print('Predicted:', decode_predictions(preds, top=3)[0])

	
	

	#network.fit(train_images, train_labels, epochs=5, batch_size=128)
	#test_loss, test_acc = network.evaluate(test_images, test_labels)
	#print('test_acc: ', test_acc)


def bilinear(inimg, x, y):

	imageSize = inimg.shape

	x = max(x, 1)
	x = min(x, imageSize[0])
	y = max(y, 1)
	y = min(y, imageSize[1])

	x0 = max(np.floor(x), 1)
	x1 = min(x0 + 1, imageSize[0])
	y0 = max(np.floor(y), 1)
	y1 = min(y0 + 1, imageSize[1])


	
	
	xZero = int(x0)
	yZero = int(y0)
	xOne = int(x1)
	yOne = int(y1)

	
	import pdb; pdb.set_trace()
	valul = inimg[xZero,yZero]
	valur = inimg[xOne,yZero]
	valll = inimg[xZero,yOne]
	vallr = inimg[xOne,yOne]

	x0 = np.floor(x)
	x1 =  x0 + 1
	y0 = np.floor(y)
	y1 = y0 + 1

	vala = (x - x0) * valur + (x1 - x) * valul
	valb = (x - x0) * vallr + (x1 - x) * valll

	value = (y - y0) * valb + (y1 - y)*vala

	return value




def affineWarp(inputImage):

	inputShape = inputImage.shape
	newImage = np.zeros((inputShape[0], inputShape[1], 3), np.uint8)

	a = 0.02
	b = 0.01
	c = 10
	d = 0.01
	e = -0.02
	f = 5
	

	for i in range(0, inputShape[0]):
		for j in range(0, inputShape[1]):
			
			uu = (a * i) + (b * j) + c
			vv = (d * i) + (e * j) + f
			newImage[j,i] = bilinear(inputImage, j + vv, i + uu)
			

	plt.imshow(newImage)
	plt.show()


def problem1():

	print "Hello"

	
	img = cv2.imread('obama.jpg')
	
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

	
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	affineWarp(img)







if __name__ == '__main__':
   
    problem1()
    #problem3()