import cv2 as cv

img = cv.imread(r"C:\Users\caretaker\Documents\hakaton\hak2023\Data\tg\images\0ac1d316-523f-4a67-8cb5-317ec4d4f41f.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.resize(gray, (1280, 720))

cv.imshow("gray", gray)
cv.waitKey(0)
