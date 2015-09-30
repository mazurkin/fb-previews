import Image
import numpy as np
from matplotlib import pyplot as plt


def img_filter(img):
    c_data = np.asarray(img.getdata()).reshape(img.size)

    c_fft = np.fft.fft2(c_data)
    c_fft_shift = np.fft.fftshift(c_fft)

    band = 8
    c_fft_shift[:band + 1, :] = 0
    c_fft_shift[32-band:, :] = 0
    c_fft_shift[:, :band + 1] = 0
    c_fft_shift[:, 32-band:] = 0

    c_ifft_shift = np.fft.ifftshift(c_fft_shift)
    c_ifft = np.fft.ifft2(c_ifft_shift)
    c_ifft_abs = np.abs(c_ifft)

    return c_ifft_abs

im = Image.open("/home/nick/Downloads/pics/pics4/supermodels_22.jpg")

im = im.resize((32, 32), Image.BICUBIC)

r, g, b = im.split()

# Plot

plt.subplot(3,2,1)
plt.imshow(r, cmap='gray')
plt.title('Original R')
plt.xticks([])
plt.yticks([])

plt.subplot(3,2,2)
plt.imshow(img_filter(r), cmap='gray')
plt.title('Reversed R')
plt.xticks([])
plt.yticks([])

plt.subplot(3,2,3)
plt.imshow(g, cmap='gray')
plt.title('Original G')
plt.xticks([])
plt.yticks([])

plt.subplot(3,2,4)
plt.imshow(img_filter(g), cmap='gray')
plt.title('Reversed G')
plt.xticks([])
plt.yticks([])

plt.subplot(3,2,5)
plt.imshow(b, cmap='gray')
plt.title('Original B')
plt.xticks([])
plt.yticks([])

plt.subplot(3,2,6)
plt.imshow(img_filter(b), cmap='gray')
plt.title('Reversed B')
plt.xticks([])
plt.yticks([])

plt.show()
