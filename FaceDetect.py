import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


def display_image(img,cmap = None,title = ""):
	plt.figure()
	plt.title(title)
	plt.imshow(img,cmap = cmap)
	plt.show()

def load_img(filename):
	img = cv2.imread(filename,0)
	return img

def detect_image(img,face_cascade):
	img_copy = img.copy()
	img_rects = face_cascade.detectMultiScale(img_copy) 
	for (x,y,w,h) in img_rects:
		cv2.rectangle(img_copy, (x,y), (x+w,y+h), (255,255,255), 10) 

	return img_copy

#first commandline argument: Filename for the image ; Second commandline argument: The XML file for training cascades
def main():

	img = load_img(sys.argv[1])
	face_cascade = cv2.CascadeClassifier(sys.argv[2])
	img_detect = detect_image(img,face_cascade)
	display_image(img_detect,cmap = "gray",title = "Face Detected")


if __name__ == '__main__':
	main()