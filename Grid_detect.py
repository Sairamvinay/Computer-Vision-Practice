import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def Grid_detect(img_data,isCircle):


	plt.figure()
	img = cv2.imread(img_data)
	img_color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	img_copy = img.copy()
	
	found = None
	corners = None
	title = str()
	shape = ()

	if isCircle == 0:
		shape = (7,7)
		found,corners = cv2.findChessboardCorners(img,shape)
		title = "Chessboard image"
	
	else:
		shape = (10,10)
		found,corners = cv2.findCirclesGrid(img, shape, cv2.CALIB_CB_SYMMETRIC_GRID)
		title = "Circle Dot image"

	
	if found:
		cv2.drawChessboardCorners(img_copy,shape,corners,found)
		plt.imshow(img_copy)
		plt.title("Grid Detection is used for "+title)
		plt.show()








def main():
	
	Grid_detect(sys.argv[1],int(sys.argv[2]))


if __name__ == '__main__':
	main()