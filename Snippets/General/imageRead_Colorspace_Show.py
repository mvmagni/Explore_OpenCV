import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt

print(f'Current directory: {os.getcwd()}')
img = cv.imread(cv.samples.findFile("starry_night.jpg"))

if img is None:
    sys.exit("Could not read the image.")


print(f'Image size is: {img.shape}')
print(f'Data type of image is: {img.dtype}')


cv.imshow("Original image", img)
#k = cv.waitKey(0)
#if k == ord("s"):
#    cv.imwrite("starry_night.png", img)

# Last parameter option for cv.imread
# cv2.IMREAD_GRAYSCALE or 0: Load the image in grayscale mode.
# cv2.IMREAD_COLOR or 1: Load the image in color mode. Any transparency in the image will be removed. This is the default.
# cv2.IMREAD_UNCHANGED or -1: Load the image unaltered; including alpha channel.
img_grayscale=cv.imread(cv.samples.findFile("starry_night.jpg"),cv.IMREAD_GRAYSCALE)
plt.title("Grayscale CV load and grayscale plot")
plt.imshow(img_grayscale,cmap='gray')
plt.show()
plt.close()

# Matplotlib expects RGB. OpenCV stores in BGR. 
# Needs to fix to display correct visuals
img_BGR_to_RGB = cv.imread(cv.samples.findFile("starry_night.jpg"),cv.IMREAD_COLOR)
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(img_BGR_to_RGB)
ax1.set_title('BGR Colormap')
ax2.imshow(img_BGR_to_RGB[:,:,::-1])
ax2.set_title('Reversed BGR Colormap(RGB)')
plt.show()
plt.close()

#Convert between colorspace
img_bgr = cv.imread(cv.samples.findFile("starry_night.jpg"),cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
plt.title("BGR load converted to RGB")
plt.imshow(img_rgb)
plt.show()
plt.close()