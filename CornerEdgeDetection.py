#Corner Detection
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

NUM_CORNERS_MAX = 64 #-1 for no limit

def Harris_detect(img_data):

	plt.figure()
	img = cv2.imread(img_data)
	img_color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	corners = cv2.cornerHarris(src = np.float32(img_gray),blockSize = 2,ksize = 3,k=0.04)
	corners = cv2.dilate(corners,None)	#get a clearer demarcation

	img_color[corners > 0.01 * corners.max()] = [255,0,0]
	plt.title("Harris Detection")
	plt.imshow(img_color)
	plt.show()

def ShiTomasi_detect(img_data):
	plt.figure()
	img = cv2.imread(img_data)
	img_color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	corners = cv2.goodFeaturesToTrack(img_gray,NUM_CORNERS_MAX,0.01,10)	#3rd parameter is quality parameter

	for i in np.int0(corners):
		x,y = i.ravel()	#i is a 2d array of just 2 values: [[x,y]]
		cv2.circle(img_color,(x,y),3,(255,0,0),-1)

	plt.title("Shi Tomasi detection")
	plt.imshow(img_color)
	plt.show()


def Canny_detect(img_data):
	plt.figure()
	img = cv2.imread(img_data)
	blurred_img = cv2.blur(img,ksize = (5,5))
	median = np.median(img)
	lower = int(max(0,0.7 * median))
	upper = int(min(255,1.3 * median))
	edges = cv2.Canny(blurred_img,threshold1 = lower,threshold2 = upper)
	plt.title("Canny Edge Detection")
	plt.imshow(edges)
	plt.show()

def main():
	if int(sys.argv[2]) == 0:
		Harris_detect(sys.argv[1])
		ShiTomasi_detect(sys.argv[1])


	else:
		Canny_detect(sys.argv[1])


if __name__ == '__main__':
	main()