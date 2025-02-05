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

imgDetecteddigits=drawGrid(imgDetecteddigits)
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
