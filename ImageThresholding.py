import cv2
import matplotlib.pyplot as plt
import numpy as numpy
import sys

WHITE_PIXELS = 255

def display_img(img,title = ""):
	fig = plt.figure(figsize = (15,15))
	ax = fig.add_subplot(111)
	plt.title(title)
	ax.imshow(img,cmap = "gray")
	plt.show()



def main():

	if(len(sys.argv) != 2):
		print("Error: need only one argument")
		exit()


	

	img = cv2.imread(sys.argv[1],0) #read in as a grey scale image
	display_img(img,"Actual image")

	thresh1 = cv2.adaptiveThreshold(img,WHITE_PIXELS,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)  #play with the blocksize (2nd last: needs to be odd) and constant to subtract from mean (last)
	display_img(thresh1,"Image with Mean C threshold")

	thresh2 = cv2.adaptiveThreshold(img,WHITE_PIXELS,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)
	display_img(thresh2,"Image with Gaussian threshold")

	thresh = cv2.addWeighted(thresh1,0.7,thresh2,0.3,gamma = 0)
	display_img(thresh,"Image with both thresholds mixed")

if __name__ == '__main__':
	main()
