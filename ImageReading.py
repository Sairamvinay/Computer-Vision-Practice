#Lecture 8: Images and Numpy
#Udemy CV
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image

pic = Image.open(sys.argv[1])

pic.show()

arr = np.asarray(pic)
print(arr.shape)

red_arr = arr.copy()
green_arr = arr.copy()
blue_arr = arr.copy()
cyan_arr = arr.copy()
yellow_arr = arr.copy()
sky_blue_arr = arr.copy()
black_arr = arr.copy()
white_arr = arr.copy()



cyan_arr[:,:,1] = 0	#MAKE GREEN channel 0
yellow_arr[:,:,2] = 0 #MAKE BLUE channel 0
sky_blue_arr[:,:,0] = 0 #MAKE RED channel 0

black_arr[:,:,:] = 0	#MAKE ALL channels 0
white_arr[:,:,:] = 255 #MAKE ALL channels full pixel
red_arr[:,:,1] = 0 #MAKE GREEN channel 0
red_arr[:,:,2] = 0 #MAKE BLUE channel 0

green_arr[:,:,0] = 0 #MAKE RED channel 0
green_arr[:,:,2] = 0 #MAKE BLUE channel 0

blue_arr[:,:,0] = 0 #MAKE RED channel 0
blue_arr[:,:,1] = 0 #MAKE GREEN channel 0


plt.figure()
plt.title("Red Channel Gray Scale")
plt.imshow(arr[:,:,0],cmap = "gray")
plt.show()
plt.figure()
plt.title("Green Channel Gray Scale")
plt.imshow(arr[:,:,1],cmap = "gray")
plt.show()
plt.figure()
plt.title("Blue Channel Gray Scale")
plt.imshow(arr[:,:,2],cmap = "gray")
plt.show()
plt.figure()
plt.title("Red Picture only")
plt.imshow(red_arr)
plt.show()
plt.figure()
plt.title("Blue Picture only")
plt.imshow(blue_arr)
plt.show()
plt.figure()
plt.title("Green Picture only")
plt.imshow(green_arr)
plt.show()
plt.figure()
plt.title("Cyan (Red and Blue) Picture only")
plt.imshow(cyan_arr)
plt.show()
plt.figure()
plt.title("Yellow (Red and Green) Picture only")
plt.imshow(yellow_arr)
plt.show()
plt.figure()
plt.title("Green and Blue Picture only")
plt.imshow(sky_blue_arr)
plt.show()
plt.figure()
plt.title("Black Picture only")
plt.imshow(black_arr)
plt.show()
plt.figure()
plt.title("White Picture only")
plt.imshow(white_arr)
plt.show()

