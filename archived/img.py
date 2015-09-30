import Image
import numpy as np

im = Image.open("/home/nick/Downloads/pics/pics4/supermodels_22.jpg")

im = im.resize((32, 32), Image.BICUBIC)

r, g, b = im.split()

data_r = np.asarray(r.getdata()).reshape(r.size)
data_g = np.asarray(r.getdata()).reshape(g.size)
data_b = np.asarray(r.getdata()).reshape(b.size)

data_r_fft = np.fft.fftshift(np.fft.fft2(data_r))
data_g_fft = np.fft.fftshift(np.fft.fft2(data_g))
data_b_fft = np.fft.fftshift(np.fft.fft2(data_b))

data_r_ifft = np.abs(np.fft.ifft2(np.fft.ifftshift(data_r_fft)))
data_g_ifft = np.abs(np.fft.ifft2(np.fft.ifftshift(data_g_fft)))
data_b_ifft = np.abs(np.fft.ifft2(np.fft.ifftshift(data_b_fft)))

data_r_mod = Image.fromarray(data_r_ifft, mode="L")
data_g_mod = Image.fromarray(data_g_ifft, mode="L")
data_b_mod = Image.fromarray(data_b_ifft, mode="L")

im1 = Image.merge("RGB", (data_r_mod, data_g_mod, data_b_mod))

im1.show(im1)
