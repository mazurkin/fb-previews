# coding=utf-8

import Image
import numpy as np
import scipy as sp
import scipy.fftpack as fp
import cv2
from matplotlib import pyplot as plt


def filter_fft(img, band=12, steps=16):
    c_data = np.array(img, dtype=np.uint8)

    c_fft = np.fft.fft2(c_data)
    c_fft_shift = np.fft.fftshift(c_fft)

    c_fft_mod = c_fft_shift

    # убираем высокочастотные компоненты
    if band > 0:
        c_fft_mod[:+band, :] = 0
        c_fft_mod[-band:, :] = 0
        c_fft_mod[:, :+band] = 0
        c_fft_mod[:, -band:] = 0

    # квантифицируем коэффициенты
    if steps > 1:
        c_abs = np.abs(c_fft_mod)
        c_ang = np.angle(c_fft_mod)

        c_fft_mod = c_abs * np.exp(1j * c_ang)

    # Возвращаем назад
    c_ifft_shift = np.fft.ifftshift(c_fft_mod)
    c_ifft = np.fft.ifft2(c_ifft_shift)
    c_ifft_abs = np.abs(c_ifft)
    c_ifft_int = c_ifft_abs.astype(np.uint8)

    return c_ifft_int

im = Image.open("/home/nick/Downloads/pics/pics4/supermodels_22.jpg")
im = im.resize((42, 42), Image.ANTIALIAS)

im_data = np.array(im, dtype=np.uint8)
im_data = cv2.GaussianBlur(im_data, (19, 19), 0)
# im_data = cv2.blur(im_data, (5, 5))
im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(im_data)
r_t, g_t, b_t = filter_fft(r, 17, 16), filter_fft(g, 17, 16), filter_fft(b, 17, 16)

merged = cv2.merge((r_t, g_t, b_t))
merged = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
cv2.imwrite('merged.png', merged)

# Plot

plt.subplot(4, 2, 1)
plt.imshow(r, cmap='gray')
plt.title('Original R')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 2)
plt.imshow(r_t, cmap='gray')
plt.title('Reversed R')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 3)
plt.imshow(g, cmap='gray')
plt.title('Original G')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 4)
plt.imshow(g_t, cmap='gray')
plt.title('Reversed G')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 5)
plt.imshow(b, cmap='gray')
plt.title('Original B')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 6)
plt.imshow(b_t, cmap='gray')
plt.title('Reversed B')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 7)
plt.imshow(im)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 8)
plt.imshow(merged)
plt.title('Merged')
plt.xticks([])
plt.yticks([])

plt.show()
