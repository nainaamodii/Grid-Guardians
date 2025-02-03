## FUNCTION FOR MAKING GRIDS ON INTERMEDIATE IMG
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


