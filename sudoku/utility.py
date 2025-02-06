## FUNCTION FOR MAKING GRIDS ON INTERMEDIATE IMG
import cv2
import numpy as np


def drawGrid(img):
    secW= int(img.shape[1]/9)
    secH=int(img.shape[0]/9)
    for i in range(9):
        pt1=(0,secH*i)
        pt2 = (img.dhape[1], secH*i)
        pt3= (secW*i, 0)
        pt4= (secW*i, img.shape[0])
        cv2.line(img, pt1, pt2, (255,255,0),2)
        cv2.line(img, pt3,pt4,(255, 255,0),2)
    return img

#1.processing image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgThreshold = cv2.adaptive(imgBlur,255,1,1,11,2)
    return imgThreshold

#reorder points for warp perspective
def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeroes((4,1,2),dtype=np.int32)
    add=myPoints.sum(1)
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2]=myPoints[np.argmax(diff)]
    return myPointsNew

#3.biggest contour
def biggestContour(contours):
    biggest=np.array([])
    max_area =0
    for i in contours:
        area=cv2.contourArea(i)
        if area>50:
            peri=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            if area>max_area and len(approx)==4:
                biggest=approx
                max_area=area
    return biggest,max_area

#6. stack all images in one window
def stackImages(imgArray,scale):
    rows= len(imgArray)
    cols=len(imgArray[0])
    rowsAvailable= isinstance(imgArray[0],list)
    width = imgArray[0][0].shape[1]
    height=imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                if len(imgArray[x][y].shape)==2:imgArray[x][y]=cv2.cvtColor(imgArray[x][y],cv2)
        imageBlank=np.zeroes((height,width,3),np.uint8)
        hor=[imageBlank]*rows
        hor_con=[imageBlank]*rows
        for x in range(0,rows):
            hor[x]=np.hstack(imgArray[x])
            hor_con[x]=np.concatenate(imgArray[x])        
        ver=np.hstack(imgArray)
        ver_con=np.concatenate(hor)
    else:
        for x in range(0,rows):
            imgArray[x]=cv2.resize(imgArray[x],(0,0),None,scale,scale)   
            if len(imgArray[x].shape)==2:imgArray[x]=cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
        hor=np.hstack(imgArray)
        hor_con=np.concatenate(imgArray)
        ver=hor
    return ver
