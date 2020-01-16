import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import classification_report

EPOCHS = 25
BATCH_SIZE = 3000

#to reverse color mapping use gray_r for cmap
def load_data(which_data = 0,to_scale = False):
	if which_data == 0:
		(X_train,Y_train),(X_test,Y_test) = keras.datasets.mnist.load_data() #a data set of different digits handwritten scanned images
		X_train = X_train.reshape(X_train.shape + (1,))
		X_test = X_test.reshape(X_test.shape + (1,))
		
		

	else:
		(X_train,Y_train),(X_test,Y_test) = keras.datasets.cifar10.load_data() #CIFAR is collection of different objects: airplanes,cats dogs and so on.
		#no need to reshape X,Y here, all are color images


	if (to_scale):
		X_train = X_train / 255.0
		X_test = X_test / 255.0

	return (X_train,Y_train,X_test,Y_test)

def CNN(input_shape = (28,28,1),output_shape = 10,num_conv_pool = 1,filters = 32,activation = "relu",optimizer = "rmsprop",loss = "categorical_crossentropy",hidden_layers = 1,hidden_neurons = 128):
	model = keras.models.Sequential()
	for i in range(num_conv_pool):
		model.add(keras.layers.Conv2D(filters=filters, kernel_size=(4,4),input_shape=input_shape, activation=activation)) #Conv 2D layer to convolute the image
		model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # a max pooling to subsample
	

	model.add(keras.layers.Flatten())	#convert 2D to 1D for the next layers

	for i in range(hidden_layers):
		model.add(keras.layers.Dense(hidden_neurons,activation = activation))
	
	model.add(keras.layers.Dense(output_shape,activation = "softmax"))
	model.compile(loss = loss,optimizer = optimizer,metrics = ["acc"])
	
	model.summary()

	return model

def main():
	X_train,Y_train,X_test,Y_test = load_data(which_data = 0,to_scale = True)
	
	Y_traincat = keras.utils.to_categorical(Y_train,10)
	Y_testcat = keras.utils.to_categorical(Y_test,10)

	print(X_train.shape)
	print(Y_traincat.shape)
	print(X_test.shape)
	print(Y_testcat.shape)


	model = CNN(input_shape = X_train.shape[1:],output_shape = 10)
	model.fit(X_train,Y_traincat,epochs = EPOCHS,batch_size = BATCH_SIZE)
	print("Testing Accuracy",model.evaluate(X_test,Y_testcat)[1] * 100.00," %")

	test_pred = model.predict_classes(X_test)
	print(classification_report(Y_test,test_pred))


	


if __name__ == '__main__':
	main()