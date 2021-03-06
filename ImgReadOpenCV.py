import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

while True:

    cv2.imshow('Puppy',img)

    # EXPLANATION FOR THIS LINE OF CODE:
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
    
    # IF we've waited at least 1 ms AND we've pressed the Esc; can also use 0xFF == ord('q')
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
