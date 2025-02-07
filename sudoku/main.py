import numpy as np
import cv2
from sudokualgo import *

from utility import *
pathImage=r"E:\X\SID\sudoku_1.png"
from tensorflow.keras.models import load_model

# loading model
model=initializePredictionModel()


# PREPARING THE IMAGE
widthImg=450
heightImg=450
img=cv2.imread(pathImage)
img=cv2.resize(img, (widthImg,heightImg))
imgBlank= np.zeros((heightImg,widthImg,3), np.uint8) # creating blank image for debugging

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
    imgWrapColoured= cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits=imgBlank.copy()
    imgWrapColoured=cv2.cvtColor(imgWrapColoured, cv2.COLOR_BGR2GRAY)

   

    # SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    imgSolvedDigits= imgBlank.copy()
    boxes= splitBoxes(imgWrapColoured)
    
    numbers=getPrediction(boxes,model)
    # numbers= [7,8,0,4,0,0,1,2,0,6,0,0,0,7,5,0,0,9,0,0,0,6,0,1,0,7,8,0,0,7,0,4,0,2,6,0,0,0,1,0,5,0,9,3,0,9,0,4,0,6,0,0,0,5,0,7,0,3,0,0,0,1,2,1,2,0,0,0,7,4,0,0,0,4,9,2,0,6,0,0,7]
    
    imgDetectedDigits=displayNumbers(imgDetectedDigits, numbers, color=(52,132,197))
   
   

    numbers=np.asarray(numbers)
    posArray= np.where(numbers>0, 0,1)

    # print(posArray)   #TESTING

## FINDING SOLUTION ON BOARD

    board = np.array(numbers).reshape(9, 9)

    # print_sudoku(board)
    # print("----------------------")
    try:
        solve_sudoku(board)
    except:
        pass
    print_sudoku(board)  
    flatlist=[]
    for sublist in board:
        for item in sublist:
            flatlist.append(item)
    solvedNumbers=flatlist*posArray
    imgSolvedNumbers=displayNumbers(imgSolvedDigits,solvedNumbers)

   

##overlay Solution(TO OVERLAY THE SOLUTION ON SAME IMAGE)

#     #preparing points for wrap
    # pts2=np.float32(biggest)
    # pts1 = np.float32([[0,0], [widthImg,0], [0,heightImg], [widthImg,heightImg]])

#     # GER
    # matrix=cv2.getPerspectiveTransform(pts1,pts2)
    # imgInvWrapColoured=img.copy()
    # imgInvWrapColoured = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))

    

#     # overwritting on ques image
    # inv_perspective = cv2.addWeighted(imgInvWrapColoured, 1, img, 0.5, 1)

   
#     # Display the result
    # cv2.imshow("Solved Sudoku", inv_perspective)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# # NUMBER DATA FOR TESTING
# numbers=[0,7,0,0,0,0,0,4,3,0,4,0,0,0,9,6,1,0,8,0,0,6,3,4,9,0,0,0,9,4,0,5,2,0,0,0,3,5,8,4,6,0,0,2,0,0,0,0,8,0,0,5,3,0,0,8,0,0,7,0,9,1,9,0,2,1,0,0,0,0,5,0,0,7,0,4,0,8,0,2]

# numbers= [7,8,0,4,0,0,1,2,0,6,0,0,0,7,5,0,0,9,0,0,0,6,0,1,0,7,8,0,0,7,0,4,0,2,6,0,0,0,1,0,5,0,9,3,0,9,0,4,0,6,0,0,0,5,0,7,0,3,0,0,0,1,2,1,2,0,0,0,7,4,0,0,0,4,9,2,0,6,0,0,7]


#   #DISPLAYIN SOLUTION IMAGE+SAVING IMAGE
#   #BONUS(WOC specific)
    sand_color = ( 128, 178, 194)
    puzzleSol=np.full((450, 450, 3), sand_color, dtype=np.uint8)
    puzzleSol=displayNumbers(puzzleSol, flatlist, color=(52,132,197))
    puzzleSol=drawGrid(puzzleSol)
    cv2.imwrite("PuzzleSoln.png", puzzleSol)

    cv2.imshow("Solved Sudoku", puzzleSol)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






