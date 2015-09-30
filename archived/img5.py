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
        c_fft_mod[:band+1, :] = 0
        c_fft_mod[-band:, :] = 0
        c_fft_mod[:, :band+1] = 0
        c_fft_mod[:, -band:] = 0

    # квантифицируем коэффициенты
    if steps > 0:
        c_fft_mod_r_step = c_fft_mod.real.max() / (steps - 1)
        c_fft_mod_i_step = c_fft_mod.imag.max() / (steps - 1)

        if c_fft_mod_r_step != 0:
            c_fft_mod.real = np.multiply(c_fft_mod_r_step, np.round(c_fft_mod.real / c_fft_mod_r_step))
        if c_fft_mod_i_step != 0:
            c_fft_mod.imag = np.multiply(c_fft_mod_i_step, np.round(c_fft_mod.imag / c_fft_mod_i_step))

    # Возвращаем назад
    c_ifft_shift = np.fft.ifftshift(c_fft_mod)
    c_ifft = np.fft.ifft2(c_ifft_shift)
    c_ifft_abs = np.abs(c_ifft)
    c_ifft_int = c_ifft_abs.astype(np.uint8)

    return c_ifft_int

im = Image.open("/home/nick/Downloads/pics/pics4/supermodels_22.jpg")
im = im.resize((42, 42), Image.ANTIALIAS)

im_data = np.array(im, dtype=np.uint8)
im_data = cv2.GaussianBlur(im_data, (9, 9), 0)
im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2HLS)

h, l, s = cv2.split(im_data)
h_t, l_t, s_t = filter_fft(h, 17, 16), filter_fft(l, 17, 16), filter_fft(s, 17, 16)

merged = cv2.merge((h_t, l_t, s_t))
merged = cv2.cvtColor(merged, cv2.COLOR_HLS2BGR)
cv2.imwrite('merged.png', merged)

# Plot

plt.subplot(4, 2, 1)
plt.imshow(h, cmap='gray')
plt.title('Original H')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 2)
plt.imshow(h_t, cmap='gray')
plt.title('Reversed H')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 3)
plt.imshow(l, cmap='gray')
plt.title('Original L')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 4)
plt.imshow(l_t, cmap='gray')
plt.title('Reversed L')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 5)
plt.imshow(s, cmap='gray')
plt.title('Original S')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 6)
plt.imshow(s_t, cmap='gray')
plt.title('Reversed S')
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
