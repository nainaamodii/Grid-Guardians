import numpy as np
import cv2

from utility import *
pathImage=r"E:\X\SID\sudoku_17.png"

# PREPARING THE IMAGE
widthImg=450
heightImg=450
img=cv2.imread(pathImage)
img=cv2.resize(img, (widthImg,heightImg))
# imgBlank=np.zeros(heightImg,widthImg) # creating blank image for debugging

imgThreshold=drawGrid
