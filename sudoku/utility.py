## FUNCTION FOR MAKING GRIDS ON INTERMEDIATE IMG
import cv2
import numpy as np

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





