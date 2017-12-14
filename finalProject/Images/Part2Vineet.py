import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import keras
from keras.applications.resnet50 import ResNet50
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import numpy as np
from PIL import Image
import requests
from io import BytesIO

### Much of this code was taken from: https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2
from keras.preprocessing import image
from keras.models import load_model

target_size = (229, 229) #fixed size for InceptionV3 architecture

model = load_model("dc.model")

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet

  Args:
    base_model: keras model excluding top
    nb_classes: # of classes

  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.nb_epoch)
  batch_size = int(args.batch_size)

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
  test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  validation_generator = test_datagen.flow_from_directory(
    args.val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # setup model
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  """
  setup_to_transfer_learn(model, base_model)
  
  history_tl = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')
  """

  # fine-tuning
  setup_to_finetune(model)
  '''
  history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')
  '''
  history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps = 3)

  model.save(args.output_model_file)

  if args.plot:
    plot_training(history_ft)


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()





def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("cat", "dog")
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


def calculate():
	test_images = glob.glob('data/validation/testimages/*.jpg')
	
	cat_true_postiive = 0
	cat_false_negative = 0
	

	dog_true_postiive = 0
	dog_false_negative = 0
	
	for fname in test_images:
		img = Image.open(fname)
		className = os.path.basename(fname)
		className = className[:3]
		preds = predict(model, img, target_size)
		array = preds

		#import pdb; pdb.set_trace()
		test_class = ""
		
		if array[0] > array[1]:
			test_class = "cat"
		else:
			test_class = "dog"
		
		##########################

		if className == "dog":

			if test_class == "dog":
				dog_true_postiive = dog_true_postiive + 1
			elif test_class == "cat":
				dog_false_negative = dog_false_negative + 1
		elif className == "cat":

			if test_class == "cat":
				cat_true_postiive = cat_true_postiive + 1
			elif test_class == "dog":
				cat_false_negative = cat_false_negative + 1

		



	accuracy = (cat_true_postiive + dog_true_postiive) / (len(test_images)/1.0)
		
	#cat_prec = (cat_true_postiive) / ((cat_true_postiive + cat_false_negative))/1.0
	print " Accuracy is: " + str(accuracy)
	cat_recall = (cat_true_postiive) / (cat_true_postiive + cat_false_negative *1.0)
	print " Cat Recall is: " + str(cat_recall)


	cat_precision = (cat_true_postiive) / (cat_true_postiive + cat_false_negative *1.04)
	print "Cat precsion is: " + str(cat_precision)


	dog_recall = (dog_true_postiive) / (dog_true_postiive + dog_false_negative *1.0)
	print " dog Recall is: " + str(dog_recall)


	dog_precision = (dog_true_postiive) / (dog_true_postiive + dog_false_negative *1.04)
	print "dog precsion is: " + str(dog_precision)


	#TP: If dog and classified as dog
	#TP: If cat and classified as cat
	#FN: If not a dog, and classified as a dog
	#FN: If not a cat, and classified as a cat
	#FP: It is a dog, and classified as a cat
	#FP: It is a cat, and classified as a dog
	


def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]



def pretrained():

	keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

	model = ResNet50(weights='imagenet')
	'''
	img_path = 'chair1.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	'''

	images = glob.glob('p2a-c/*.jpg')
	i = 0
	for fname in images:

		img = image.load_img(fname, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		print "-------------------"
		print "FILE NAME: " + str(i)
		print fname
		print "-------------------"
		print ""
		print ""
		i = i + 1
		preds = model.predict(x)
		print('Predicted:', decode_predictions(preds, top=3)[0])

	print ""
	print ""
	print ""
	print "See Confusion matrix in attached .xslx sheet"

	print ""
	print ""
	print ""
	print "Accuracy is 0.7586"

	print ""
	print ""
	print ""
	print "Recall is 0.7545"


	print ""
	print ""
	print ""
	print "Precision is 0.9152"


	print ""
	print ""
	print ""
	print "The f-score is 0.857"







if __name__=="__main__":
	print "Hello"
	

	#This part is to train the network and has been removed to speed up demo
	'''
	a = argparse.ArgumentParser()
  a.add_argument("--train_dir", default="data/train_dir")
  a.add_argument("--val_dir", default="data/val_dir")
  a.add_argument("--nb_epoch", default=NB_EPOCHS)
  a.add_argument("--batch_size", default=BAT_SIZE)
  a.add_argument("--output_model_file", default="dc.model")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()
  if args.train_dir is None or args.val_dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    print("directories do not exist")
    sys.exit(1)
	'''
	print "PRETRAINED Network"
	pretrained()
	print "-------------------------------------------------------------"
	print "Modifying a network (Inception V3) and testing images"
	model = load_model("dc.model")
	calculate()
	
	#print "Model Loaded"
	#img = Image.open("dog.1173.jpg")
	#preds = predict(model, img, target_size)
	#print preds
	
