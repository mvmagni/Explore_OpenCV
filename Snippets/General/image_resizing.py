import cv2 as cv
import matplotlib.pyplot as plt

# Image resizing
img = cv.imread(cv.samples.findFile("starry_night.jpg"), cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#cv.imshow("original image", img)

plt.figure(figsize=[18, 5])


plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(img_rgb)


plt.subplot(1, 4, 2)  # row 1, column 3, count 1

cropped_region = img_rgb[200:400, 100:300]
resized_img_5x = cv.resize(cropped_region, None, fx=5, fy=5)
plt.imshow(resized_img_5x)
plt.title("Resize Cropped Image with Scale 5X")

width = 200
height = 300
dimension = (width, height)
resized_img = cv.resize(img_rgb, dsize=dimension, interpolation=cv.INTER_AREA)

plt.subplot(1, 4, 3)
plt.imshow(resized_img)
plt.title("Resize Image with Custom Size")

desired_width = 500
aspect_ratio = desired_width / cropped_region.shape[1]
desired_height = int(cropped_region.shape[0] * aspect_ratio)
dim = (desired_width, desired_height)
resized_cropped_region = cv.resize(cropped_region, dsize=dim, interpolation=cv.INTER_AREA)

plt.subplot(1, 4, 4)
plt.imshow(resized_cropped_region)
plt.title("Keep Aspect Ratio - Resize Cropped Region")
plt.show()