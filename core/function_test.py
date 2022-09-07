import cv2 as cv
import os

print(f'{os.getcwd()}')
alpha = 0.5
# [load]
src1 = cv.imread('Explore_OpenCV/resources/background.jpg')
src2 = cv.imread('Explore_OpenCV/resources/HUD.jpg')
# [load]
if src1 is None:
    print("Error loading src1")
    exit(-1)
elif src2 is None:
    print("Error loading src2")
    exit(-1)
# [blend_images]
beta = (1.0 - alpha)
dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
# [blend_images]
# [display]
cv.imshow('dst', dst)
cv.waitKey(0)
# [display]
cv.destroyAllWindows()