# coding=utf-8

import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt


def filter_fft(img, name):
    c_data = np.array(img, dtype=np.uint8)

    # FFT
    c_fft = np.fft.fft2(c_data)

    # разделяем гармоники на амплитуду и фазу
    c_abs = np.abs(c_fft)
    c_ang = np.angle(c_fft)

    # вычисляем порог ниже которого находится большая часть гармоник (по амплитуде)
    abs_limit = np.percentile(c_abs, 97)

    # сбрасываем все гармоники ниже этого порога
    m = (c_abs <= abs_limit)
    c_abs[m] = 1
    c_ang[m] = 0

    # квантизируем амплитуду
    c_abs = np.where(c_abs > 1, np.log(c_abs), 0)
    c_abs_steps = 16
    c_abs_step = c_abs.max() / (c_abs_steps - 1)
    c_abs_data = np.round(c_abs / c_abs_step).astype(np.uint8)

    # квантизируем фазу (-PI..+PI)
    c_ang_steps = 16
    c_ang_step = 2 * np.pi / (c_ang_steps - 1)
    c_ang_data = np.round(c_ang / c_ang_step).astype(np.int8)

    # сохраняем
    c_abs_data.tofile(name + 'abs.dat')
    c_ang_data.tofile(name + 'ang.dat')

    # восстановление амплитуды
    c_abs = c_abs_step * c_abs_data
    c_abs = np.where(c_abs > 0, np.exp(c_abs), 0)

    # восстановление угла
    c_ang = c_ang_step * c_ang_data

    # возращаем к комплексному виду
    c_fft_rst = np.multiply(c_abs, np.exp(1j * c_ang))

    # IFFT
    c_ifft = np.fft.ifft2(c_fft_rst)
    c_ifft_abs = np.abs(c_ifft)
    c_ifft_int = c_ifft_abs.astype(np.uint8)

    print("nzero={0}".format(np.count_nonzero(c_abs_data)))

    return c_ifft_int

im_src = Image.open('/home/nick/Pictures/screenshots/Selection_247.jpg')
# im = Image.open('/home/nick/Downloads/pics/pics4/gisele_bundchen.jpg')
# im = Image.open('/home/nick/Downloads/pics/pics4/670x503_Quality100_14185-1600.jpg')
# im = Image.open('/home/nick/Downloads/pics/pics4/supermodels_22.jpg')

im_dim = im_src.resize((42, 42), Image.ANTIALIAS)
im_dat = np.array(im_dim, dtype=np.uint8)
im_dat = cv2.GaussianBlur(im_dat, (21, 21), 0)
im_dat = cv2.cvtColor(im_dat, cv2.COLOR_BGR2HLS)

s1_s, s2_s, s3_s = cv2.split(im_dat)
s1_t, s2_t, s3_t = filter_fft(s1_s, 's1'), filter_fft(s2_s, 's2'), filter_fft(s3_s, 's3')

merged = cv2.merge((s1_t, s2_t, s3_t))
merged = cv2.cvtColor(merged, cv2.COLOR_HLS2BGR)

merged_inc = cv2.resize(merged, im_src.size)
merged_inc = cv2.GaussianBlur(merged_inc, (99, 99), 0)

merged_save = cv2.cvtColor(merged_inc, cv2.COLOR_BGR2RGB)
cv2.imwrite("merged.jpg", merged_save)

# Plot 2

plt.subplot(1, 2, 1)
plt.imshow(im_src)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(merged_inc)
plt.title('Merged')
plt.xticks([])
plt.yticks([])

plt.show()

# Plot 1

plt.subplot(4, 2, 1)
plt.imshow(s1_s, cmap='gray')
plt.title('Original S1')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 2)
plt.imshow(s1_t, cmap='gray')
plt.title('Reversed S1')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 3)
plt.imshow(s2_s, cmap='gray')
plt.title('Original S2')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 4)
plt.imshow(s2_t, cmap='gray')
plt.title('Reversed S2')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 5)
plt.imshow(s3_s, cmap='gray')
plt.title('Original S3')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 6)
plt.imshow(s3_t, cmap='gray')
plt.title('Reversed S3')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 7)
plt.imshow(im_dim)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 8)
plt.imshow(merged)
plt.title('Merged')
plt.xticks([])
plt.yticks([])

plt.show()

