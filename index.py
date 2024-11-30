import cv2 
import numpy as np 

img1 = cv2.imread( './img_query_1.jpg')

cv2.imshow('img1', img1 )

cv2.waitKey(0)
cv2.destroyAllWindows()
