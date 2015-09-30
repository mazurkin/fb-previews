# coding=utf-8

import Image
import numpy as np
import scipy as sp
import scipy.fftpack as fp
import cv2
from matplotlib import pyplot as plt


def filter_fft(img, band=12, step=0.0625):
    c_data = np.array(img, dtype=np.uint8)

    c_fft = np.fft.fft2(c_data)
    c_fft_shift = np.fft.fftshift(c_fft)

    c_fft_mod = c_fft_shift

    # убираем высокочастотные компоненты
    c_fft_mod[:band, :] = 0
    c_fft_mod[-band:, :] = 0
    c_fft_mod[:, :band] = 0
    c_fft_mod[:, -band:] = 0

    # огрубляем коэффициенты
    c_fft_max_r = c_fft_mod.real.max()
    c_fft_max_i = c_fft_mod.imag.max()

    if c_fft_max_r != 0:
        c_fft_mod.real /= c_fft_max_r
    if c_fft_max_i != 0:
        c_fft_mod.imag /= c_fft_max_i

    c_fft_mod.real = np.multiply(step, np.round(c_fft_mod.real / step))
    c_fft_mod.imag = np.multiply(step, np.round(c_fft_mod.imag / step))

    c_fft_mod.real *= c_fft_max_r
    c_fft_mod.imag *= c_fft_max_i

    # Возвращаем назад
    c_ifft_shift = np.fft.ifftshift(c_fft_mod)
    c_ifft = np.fft.ifft2(c_ifft_shift)
    c_ifft_abs = np.abs(c_ifft)
    c_ifft_int = c_ifft_abs.astype(np.uint8)

    img_f = Image.fromarray(c_ifft_int, mode="L")

    return img_f


def filter_dcs(img):
    c_data = np.array(img, dtype=np.float)

    c_dct = fp.dct(c_data, type=3, norm='ortho')

    # убираем высокочастотные компоненты
    band = 8
    c_dct[:, 16:] = 0

    # огрубляем коэффициенты
    c_dct = np.multiply(32, np.round(c_dct / 32))

    c_idct = fp.idct(c_dct, type=3, norm='ortho')
    c_idct_rnd = np.round(c_idct)
    c_idct_int = c_idct_rnd.astype(np.uint8)

    img_f = Image.fromarray(c_idct_int, mode="L")

    return img_f

im = Image.open("/home/nick/Downloads/pics/pics4/supermodels_22.jpg")
im = im.resize((32, 32), Image.ANTIALIAS)

r, g, b = im.split()
filter_img = filter_fft
r_t, g_t, b_t = filter_img(r), filter_img(g), filter_img(b)

merged = Image.merge("RGB", (r_t, g_t, b_t))
merged.save("merged.png")

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
