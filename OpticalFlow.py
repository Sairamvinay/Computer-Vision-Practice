import cv2
import numpy as np
import matplotlib.pyplot as plt

def LK_OPTFLOW():
    # Parameters for ShiTomasi corner detection (good features to track paper)
    corner_track_params = dict(maxCorners = 10,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )



    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (200,200),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03))


    # Capture the video
    cap = cv2.VideoCapture(0)

    # Grab the very first frame of the stream
    ret, prev_frame = cap.read()

    # Grab a grayscale image (We will refer to this as the previous frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Grabbing the corners
    prevPts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **corner_track_params)

    # Create a matching mask of the previous frame for drawing on later
    mask = np.zeros_like(prev_frame)


    while True:
        
        # Grab current frame
        ret,frame = cap.read()
        
        # Grab gray scale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the Optical Flow on the Gray Scale Frame
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)
        
        # Using the returned status array (the status output)
        # status output status vector (of unsigned chars); each element of the vector is set to 1 if
        # the flow for the corresponding features has been found, otherwise, it is set to 0.
        good_new = nextPts[status==1]
        good_prev = prevPts[status==1]
        
        # Use ravel to get points to draw lines and circles
        for i,(new,prev) in enumerate(zip(good_new,good_prev)):
            
            x_new,y_new = new.ravel()
            x_prev,y_prev = prev.ravel()
            
            # Lines will be drawn using the mask created from the first frame
            mask = cv2.line(mask, (x_new,y_new),(x_prev,y_prev), (0,255,0), 3)
            
            # Draw red circles at corner points
            frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
        
        # Display the image along with the mask we drew the line on.
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
       
        # Now update the previous frame and previous points
        prev_gray = frame_gray.copy()
        prevPts = good_new.reshape(-1,1,2)
        
        
    cv2.destroyAllWindows()
    cap.release()


def FB_OPTFLOW():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()

    # Get gray scale image of first frame and make a mask in HSV color
    prvsImg = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    hsv_mask = np.zeros_like(frame1)
    hsv_mask[:,:,1] = 255

    while True:
        ret, frame2 = cap.read()
        nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        
        # Check out the markdown text above for a break down of these paramters, most of these are just suggested defaults
        flow = cv2.calcOpticalFlowFarneback(prvsImg,nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        
        # Color the channels based on the angle of travel
        # Pay close attention to your video, the path of the direction of flow will determine color!
        mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1],angleInDegrees=True)
        hsv_mask[:,:,0] = ang/2
        hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        
        # Convert back to BGR to show with imshow from cv
        bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        # Set the Previous image as the next iamge for the loop
        prvsImg = nextImg

        
    cap.release()
    cv2.destroyAllWindows()

def main():
    #LK_OPTFLOW()
    FB_OPTFLOW()


if __name__ == '__main__':
    main()


