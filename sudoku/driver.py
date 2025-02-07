import numpy as np
import cv2
from tensorflow
from utility import *
pathImage=r"E:\X\SID\sudoku_17.png"
from tensorflow.keras.models import load_model




# PREPARING THE IMAGE
widthImg=450
heightImg=450
img=cv2.imread(pathImage)
img=cv2.resize(img, (widthImg,heightImg))
imgBlank=np.zeros((heightImg,widthImg,3), np.uint8) # creating blank image for debugging

imgThreshold=preProcess(img)

# FINDING CONTOURS
contours, hierarchy=cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# FINDING BIGGEST CONTOUR (SODUKU)
biggest, maxArea= biggestContour(contours)
if biggest.size != 0:
    biggest=reorder(biggest)

    # preparing points for wrap
    pts1= np.float32(biggest)
    pts2= np.float32([[0,0], [widthImg,0],[0,heightImg],[widthImg, heightImg]])

    matrix= cv2.getPerspectiveTransform(pts1, pts2)
    imgWrapColoured= cv2.wrapPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits= imgBlank.copy()
    imgWrapColoured=cv2.cvtColor(imgWrapColoured, cv2.COLOR_BGR2GRAY)

    # SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    imgSolvedDigits= imgBlank.copy()
    boxes= splitBoxes(imgWrapColoured)
    
    numbers=getPrediction(boxes,model)
    imgDetectedDigits=displayNumbers(imgDetectedDigits, numbers, colour=(255,0,255))

    numbers=np.asarray(numbers)
    posArray= np.where(numbers>0, 0,1)

    print(posArray)

## FINDING SOLUTION ON BOARD(TO BE DONE BY AADYA)

    # board=np.array_split(numbers, 9)
    # print(board)
    # try:
        # SudokuSolver.Solve(board)
    # except:
        # pass
    # print(board)

##overlay Solution
    import cv2
    import numpy as np

    #preparing points for wrap
    pts2=np.float32(biggest)
    pts1 = np.float32([[0,0], [widthImg,0], [0,heightImg], [widthImg,heightImg]])

    # GER
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgInvWrapColoured=img.copy()
    imgInvWrapColoured = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))

    # overwritting onques image
    inv_perspective = cv2.addWeighted(imgInvWrapColoured, 1, img, 0.5, 1)

    imgDetecteddigits=drawGrid(imgDetectedDigits)
    imgSolvedDigits=drawGrid(imgSolvedDigits)

    # Display the result
    cv2.imshow("Solved Sudoku", inv_perspective)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    ## idea for extra points:
    '''since our theme of event is egypt so rather than overlaying the image on original image 
        we can display one more img in egyptian style format '''

    ''' read ur problem solution again and decorate ur project with problem statement deco'''

    # overlay solution part for extra points
