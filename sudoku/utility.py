## FUNCTION FOR MAKING GRIDS ON INTERMEDIATE IMG
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def drawGrid(img):
    secW= int(img.shape[1]/9)
    secH=int(img.shape[0]/9)
    for i in range(9):
        pt1=(0,secH*i)
        pt2 = (img.shape[1], secH*i)
        pt3= (secW*i, 0)
        pt4= (secW*i, img.shape[0])
        cv2.line(img, pt1, pt2, (19,69,139),2)
        cv2.line(img, pt3,pt4,(19,69,139),2)
    return img

# loading the model 
def initializePredictionModel():
    model = load_model('digit_recognition.keras')
    return model


#1.processing image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    # imgThreshold = cv2.adaptive(imgBlur,255,1,1,11,2)
    # Apply global thresholding
    # imgThreshold = cv2.threshold(imgBlur, 127, 255, cv2.THRESH_BINARY)
    # return imgThreshold
    return imgBlur

#reorder points for warp perspective
def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),dtype=np.int32)
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

#4. get predictions on all images
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
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        print(classIndex, probabilityValue)

        # save to result
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

# to display the solution on the image 
# Function to display numbers
def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = img.shape[1] // 9  # Width of each cell
    secH = img.shape[0] // 9  # Height of each cell

    if len(numbers) != 81:
        raise ValueError("The 'numbers' list must contain exactly 81 elements.")

    for y in range(9):
        for x in range(9):
            num = numbers[y * 9 + x]
            if num != 0:  # Skip empty cells
                text = str(num)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 2)[0]

                # Compute text position (centered)
                text_x = x * secW + (secW - text_size[0]) // 2
                text_y = (y + 1) * secH - (secH - text_size[1]) // 2

                # Put text on image
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)

    return img


# to split the images into 81 different boxes
def splitBoxes(img):
    rows= np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes



