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

# get predictions on all images 
def getPrediction(boxes, model):
    result = []
    for image in boxes:
        # prepare image
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        
        # get prediction
        predictions = model.predict(img)
        # classIndex = model.predict_classes(img)
        classIndex = np.argmax(predictions, axis=1)
        probabilityValue = np.amax(predictions)
        print(classIndex, probabilityValue)
