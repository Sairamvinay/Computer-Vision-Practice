import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

FLANN_INDEX_KDTREE = 0

def display(img,cmap = "gray",title = ""):
	fig = plt.figure(figsize = (12,10))
	ax = fig.add_subplot(111)
	plt.title(title)
	ax.imshow(img,cmap=cmap)
	plt.show()


def SIFT(full_img_data,small_img_data,Flann = False,maskNeeded = False,flags = 0):
	full = cv2.imread(full_img_data,0)
	small = cv2.imread(small_img_data,0)
	sift = cv2.xfeatures2d.SIFT_create()
	kp1,des1 = sift.detectAndCompute(small,mask = None)
	kp2,des2 = sift.detectAndCompute(full,mask = None)
	bf = None
	title = str()
	if Flann:
		title = "Flann used SIFT matcher"
		index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
		search_params = dict(checks = 50)
		bf = cv2.FlannBasedMatcher(index_params,search_params)

	else:
		title = "Regular SIFT matcher"
		bf = cv2.BFMatcher()
	
	matches = bf.knnMatch(des1,des2, k=2)

	if maskNeeded == False:	
		# Apply ratio test
		good = [] #we narrow down the matches
		for match1,match2 in matches:
			#if match1 distance is less than 75% of the match 2 distance
			if match1.distance < 0.75*match2.distance:
				good.append([match1])

		# cv2.drawMatchesKnn expects list of lists as matches.
		sift_matches = cv2.drawMatchesKnn(small,kp1,full,kp2,good,None,flags=flags)
		display(sift_matches,title = title)

	else:
		masks = [[0,0] for i in range(len(matches))]
		# ratio test
		for i,(match1,match2) in enumerate(matches):
		    if match1.distance < 0.7*match2.distance:
		        masks[i]=[1,0]

		draw_params = dict(matchColor = (0,255,0),
		                   singlePointColor = (255,0,0),
		                   matchesMask = masks,
		                   flags = flags)

		sift_matches = cv2.drawMatchesKnn(small,kp1,full,kp2,matches,None,**draw_params)
		display(sift_matches,title = title + " with Masks")


def ORB(full_img_data,small_img_data):

	full = cv2.imread(full_img_data,0)
	small = cv2.imread(small_img_data,0)
	orb = cv2.ORB_create()
	kp1,des1 = orb.detectAndCompute(small,mask = None)
	kp2,des2 = orb.detectAndCompute(full,mask = None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
	matches = bf.match(des1,des2)
	matches = sorted(matches,key = lambda x:x.distance)

	all_matches = cv2.drawMatches(small,kp1,full,kp2,matches[:50],None,flags = 2) #only lines are shown
	display(all_matches,title = "ORB Matcher")


def main():
	ORB(sys.argv[1],sys.argv[2])
	SIFT(sys.argv[1],sys.argv[2],flags = 2)
	SIFT(sys.argv[1],sys.argv[2],Flann = True,flags=2)
	SIFT(sys.argv[1],sys.argv[2],Flann = True,maskNeeded = True,flags=0)


if __name__ == '__main__':
	main()