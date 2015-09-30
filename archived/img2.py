import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("/home/nick/Downloads/pics/pics4/supermodels_22.jpg", 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)

plt.imshow(img_back)
plt.title('result')
plt.xticks([])
plt.yticks([])
plt.show()

