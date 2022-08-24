import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image as gray scale.
img = cv.imread(cv.samples.findFile("gradient.png"),cv.IMREAD_GRAYSCALE)
# Set color map to gray scale for proper rendering.
plt.imshow(img, cmap='gray')
# Print img pixels as 2D Numpy Array
print(img)
# Show image with Matplotlib
plt.title(f'Gradient grayscale')
plt.show()


print(f'Image size: {img.shape}')
# print the first pixel
print(f'First pixel: {str(img[0,0])}')
# print the white pixel to the top right corner
print(f'Top right pixel: {img[0,299]}')


# Modify a range of pixels
gr_img = img.copy()

# Modify pixel one by one
#gr_img[20,20] = 200
#gr_img[20,21] = 200
#gr_img[20,22] = 200
#gr_img[20,23] = 200
#gr_img[20,24] = 200
# ...

# Modify pixel between 20-80 pixel range
gr_img[20:150,20:80] = 200

plt.imshow(gr_img, cmap='gray')
print(gr_img)
plt.show()


# Image cropping
img_uncropped = cv.imread(cv.samples.findFile("starry_night.jpg"), cv.IMREAD_COLOR)
print(f'Uncropped image size: {img_uncropped.shape}')
img_rgb = cv.cvtColor(img_uncropped, cv.COLOR_BGR2RGB)
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(img_rgb)
ax1.set_title("Before crop")
ax2.imshow(img_rgb[200:400, 100:300])
ax2.set_title("After crop")
plt.show()

