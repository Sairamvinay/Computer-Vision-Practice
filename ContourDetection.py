import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt



def Contour_detect(img_data):

	img = cv2.imread(img_data,0)
	contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

	external_contour = np.zeros(img.shape)
	internal_contour = np.zeros(img.shape)

	plt.figure()
	for i in range(len(contours)):

		if hierarchy[0][i][-1] == -1:

			cv2.drawContours(external_contour,contours,i,255,-1) #255 is color of the image; -1 for filled in; i is for the contour to draw



	plt.title("External Contour alone")
	plt.imshow(external_contour,cmap = "gray")
	plt.show()


	plt.figure()
	plt.title("Internal Contour alone")
	for i in range(len(contours)):

		if hierarchy[0][i][-1] != -1:

			cv2.drawContours(internal_contour,contours,i,255,-1) #255 is color of the image; -1 for filled in; i is for the contour to draw

	plt.imshow(internal_contour,cmap = "gray")
	plt.show()


def main():
	Contour_detect(sys.argv[1])


if __name__ == '__main__':
	main()