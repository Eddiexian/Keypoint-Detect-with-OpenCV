from skimage import data, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
import numpy as np

im = cv2.imread('Lenna.jpg')
im = np.float32(im) / 255.0

# 計算梯度
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# 計算梯度幅度和方向
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
cv2.imshow("absolute x- gradient", gx)
cv2.imshow("absolute y-gradient", gy)
cv2.imshow("gradient magnitude", mag)
cv2.imshow("gradient direction", angle)
cv2.waitKey(0)


fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
