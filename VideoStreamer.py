import cv2
import numpy as np
import time

FRAME_COUNT = 35

pt1 = (0,0)
pt2 = (0,0)
topLeft_clicked = False
botRight_clicked = False


# mouse callback function
def draw_rectangle(event,x,y,flags,param):

    
	global pt1,pt2,topLeft_clicked,botRight_clicked
	# get mouse click
	if event == cv2.EVENT_LBUTTONDOWN:
		if topLeft_clicked == True and botRight_clicked == True:
			topLeft_clicked = False
			botRight_clicked = False
			pt1 = (0,0)
			pt2 = (0,0)

		if topLeft_clicked == False:
			pt1 = (x,y)
			topLeft_clicked = True

		elif botRight_clicked == False:
			pt2 = (x,y)
			botRight_clicked = True

        

def live_drawer():

	# Haven't drawn anything yet!

	
	cap = cv2.VideoCapture(0) 

	# Create a named window for connections
	cv2.namedWindow('Test')

	# Bind draw_rectangle function to mouse cliks
	cv2.setMouseCallback('Test', draw_rectangle) 


	while True:
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    if topLeft_clicked:
	        cv2.circle(frame, center=pt1, radius=5, color=(0,0,255), thickness=-1)
	        
	    #drawing rectangle
	    if topLeft_clicked and botRight_clicked:
	        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
	        
	        
	    # Display the resulting frame
	    cv2.imshow('Test', frame)

	    # This command let's us quit with the "q" button on a keyboard.
	    # Simply pressing X on the window won't work!
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	# When everything is done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def draw_rect():
	cap = cv2.VideoCapture(0)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	x1 = width // 4
	y1 = height // 4

	x2 = 3 * width // 4
	y2 = 3 * height // 4

	while True:

		ret,frame = cap.read()

		cv2.rectangle(frame,(x1,y1),(x2,y2),color = (0,0,255),thickness = 5)
		cv2.imshow("frame",frame)

		#press q to stop the streaming
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	cap.release()
	cv2.destroyAllWindows()




def record_video(filename="awesome_video.mp4"):
	cap = cv2.VideoCapture(0)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	writer = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'XVID'),FRAME_COUNT,(width,height))

	while True:

		ret,frame = cap.read()
		pict = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		
		#cv2.imshow("frame",pict)
		
		cv2.imshow("frame",frame)
		writer.write(frame)
		
		#press q to stop the streaming
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture and destroy the windows
	cap.release()
	writer.release()
	cv2.destroyAllWindows()


def stream_video(filename = "awesome_video.mp4"):
	
	cap = cv2.VideoCapture(filename)

	fps = FRAME_COUNT
	waiting_time = 1.0/fps

	if cap.isOpened() == False:
		print("Error: file not found or Wrong Codec Format")


	#while video is opened and running
	while cap.isOpened():

		ret,frame = cap.read()

		#frame is found
		if (ret == True):

			time.sleep(waiting_time)
			cv2.imshow("frame",frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		else:
			break

	cap.release()
	cv2.destroyAllWindows()


def main():

	live_drawer()
	#draw_rect()
	#record_video()
	#stream_video()


if __name__ == '__main__':
	main()





