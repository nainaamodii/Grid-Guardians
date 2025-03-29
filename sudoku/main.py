# MAIN WORKLFLOW
import cv2
import numpy as np
import operator
from tensorflow.keras.models import load_model
import sudoku_solver as sol  # Assuming you have a Sudoku solver function
from utility import *

# Load the trained digit classifier model
classifier = load_model("./digit_model.h5")

# Define grid parameters
marge = 4
case = 28 + 2 * marge
taille_grille = 9 * case

# change the path image for any other sudoku image ques
image_path = r"resources\sudoku_2.png"
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Unable to load image!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

# Find contours to detect the largest rectangular grid
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_grille = None
maxArea = 0

for c in contours:
    area = cv2.contourArea(c)
    if area > 25000:
        peri = cv2.arcLength(c, True)
        polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
        if area > maxArea and len(polygone) == 4:
            contour_grille = polygone
            maxArea = area

# If a Sudoku grid is found
if contour_grille is not None:
    cv2.drawContours(frame, [contour_grille], 0, (0, 255, 0), 2)

    # Get perspective transform
    points = np.vstack(contour_grille).squeeze()
    points = sorted(points, key=operator.itemgetter(1))  # Sort by y-coordinates

    if points[0][0] < points[1][0]:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[0], points[1], points[3], points[2]])
        else:
            pts1 = np.float32([points[0], points[1], points[2], points[3]])
    else:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[1], points[0], points[3], points[2]])
        else:
            pts1 = np.float32([points[1], points[0], points[2], points[3]])

    pts2 = np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [taille_grille, taille_grille]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    grille = cv2.warpPerspective(frame, M, (taille_grille, taille_grille))

    # Convert the extracted grid to grayscale and apply thresholding
    grille = cv2.cvtColor(grille, cv2.COLOR_BGR2GRAY)
    grille = cv2.adaptiveThreshold(grille, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

    cv2.imshow("Extracted Sudoku Grid", grille)

    # Recognize digits using the trained model
    grille_txt = []
    for y in range(9):
        ligne = ""
        for x in range(9):
            y2min = y * case + marge
            y2max = (y + 1) * case - marge
            x2min = x * case + marge
            x2max = (x + 1) * case - marge
            img = grille[y2min:y2max, x2min:x2max]
            x_img = img.reshape(1, 28, 28, 1)

            if x_img.sum() > 10000:  # If there is a digit present
                # prediction = classifier.predict_classes(x_img)
                prediction = np.argmax(classifier.predict(x_img), axis=-1)

                ligne += "{:d}".format(prediction[0])
            else:
                ligne += "0"

        grille_txt.append(ligne)

    print("Recognized Grid:")
    for row in grille_txt:
        print(row)

    # Solve the Sudoku puzzle
    result = sol.sudoku(grille_txt)
    print("Solved Sudoku:", result)

    if result is not None:
        # displaying and saving result
        flatlist=[]
        for sublist in result:
            for item in sublist:
                flatlist.append(item)
        
        sand_color = ( 128, 178, 194)
        puzzleSol=np.full((450, 450, 3), sand_color, dtype=np.uint8)
        puzzleSol=displayNumbers(puzzleSol, flatlist, color=(52,132,197))
        puzzleSol=drawGrid(puzzleSol)
        cv2.imwrite("PuzzleSoln.png", puzzleSol)

        cv2.imshow("Solved Sudoku", puzzleSol)

    else:
        print("No solution found.")

else:
    print("No Sudoku grid detected.")

cv2.waitKey(0)
cv2.destroyAllWindows()
