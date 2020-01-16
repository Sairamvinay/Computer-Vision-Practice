import cv2
import keras
import matplotlib.pyplot
from sklearn.metrics import classification_report
import numpy as np
import sys


EPOCHS = 100

def CNN(input_shape = (28,28,1),output_shape = 10,num_conv_pool = 1,filters = 32,activation = "relu",optimizer = "rmsprop",loss = "categorical_crossentropy",hidden_layers = 1,hidden_neurons = 128):
	model = keras.models.Sequential()
	for i in range(num_conv_pool):
		model.add(keras.layers.Conv2D(filters=filters, kernel_size=(4,4),input_shape=input_shape, activation=activation)) #Conv 2D layer to convolute the image
		model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # a max pooling to subsample
	

	model.add(keras.layers.Flatten())	#convert 2D to 1D for the next layers

	for i in range(hidden_layers):
		model.add(keras.layers.Dense(hidden_neurons,activation = activation))
	
	model.add(keras.layers.Dropout(0.5))

	model.add(keras.layers.Dense(output_shape,activation = "sigmoid"))
	model.compile(loss = loss,optimizer = optimizer,metrics = ["acc"])
	
	model.summary()

	return model


def load_images(folder_path,target_size,batch_size = 20):
	image_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


	train_img = image_gen.flow_from_directory(str(folder_path + "train"),target_size=target_size[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
	test_img = image_gen.flow_from_directory(str(folder_path + "test"),target_size=target_size[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

	return train_img,test_img


def main():
	folder_path = sys.argv[1]
	train_img,test_img = load_images(folder_path,(150,150,3))
	model = CNN(input_shape = (150,150,3),num_conv_pool = 3,filters = 128,hidden_layers = 3,hidden_neurons = 128,activation = "Adam")
	results = model.fit_generator(train_img,epochs=EPOCHS,
                              steps_per_epoch=150,
                              validation_data=test_img,
                             validation_steps=12)

	dog_file = folder_path + "test/Dog/9455.png"
	cat_file = folder_path + "test/Cat/9455.png"

	dog_img = keras.preprocessing.image.load_img(dog_file, target_size=(150, 150))
	cat_img = keras.preprocessing.image.load_img(cat_file, target_size=(150, 150))
	dog_img = keras.preprocessing.image.img_to_array(dog_img)
	cat_img = keras.preprocessing.image.img_to_array(cat_img)
	dog_img = np.expand_dims(dog_img, axis=0) #for one particular sample to convert the size of the sample
	cat_img = np.expand_dims(cat_img, axis=0)
	dog_img = dog_img/255
	cat_img = cat_img/255

	predict_dog = model.predict_classes(dog_img)
	predict_cat = model.predict_classes(cat_img)
	print('Probability that dog image is a dog is:  ',predict_dog)
	print('Probability that cat image is a cat is:  ',predict_cat)



if __name__ == '__main__':
	main()