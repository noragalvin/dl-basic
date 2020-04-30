import cv2
import numpy as np

img_path = 'training/0/img_153.jpg'
img = cv2.imread(img_path, 0)

width = 14
height = 14
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print(resized)