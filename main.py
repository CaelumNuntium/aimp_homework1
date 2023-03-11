import argparse
import math
import numpy
from astropy.io import fits
from matplotlib import pyplot as plt
from photutils import aperture


class Line(object):
    def __init__(self, k, b):
        self.k = k
        self.b = b

    def __call__(self, x):
        return self.k * x + self.b


class MyArray(object):
    def __init__(self, arr):
        self.array = arr

    def __getitem__(self, item):
        try:
            return self.array[item]
        except IndexError:
            return 0


def mean_img(fits_file, datatype='f4'):
    res = numpy.ndarray(shape=fits_file[1].data.shape, dtype=datatype)
    numpy.mean([hdu.data for hdu in fits_file if hdu.data is not None], axis=0, out=res)
    return res


def median_img(fits_file, datatype='f4'):
    res = numpy.ndarray(shape=fits_file[1].data.shape, dtype=datatype)
    numpy.median([hdu.data for hdu in fits_file if hdu.data is not None], axis=0, out=res)
    return res


def slicing(image, line, step, datatype='f4'):
    image = MyArray(image)
    shape = image.array.shape
    if 0 <= line(0) <= shape[1]:
        xl = 0
        yl = line(0)
    elif line(0) < 0:
        xl = -line.b / line.k
        yl = 0
    else:
        xl = (shape[1] - line.b) / line.k
        yl = shape[1]
    if 0 <= line(shape[0]) <= shape[1]:
        xr = shape[0]
        yr = line(shape[0])
    elif line(shape[0]) < 0:
        xr = -line.b / line.k
        yr = 0
    else:
        xr = (shape[1] - line.b) / line.k
        yr = shape[1]
    n = math.floor(math.sqrt((xr - xl) ** 2 + (yr - yl) ** 2) / step)
    dx = step * math.sqrt(1 / (line.k ** 2 + 1))
    res = numpy.ndarray(shape=(2, n), dtype=datatype)
    for i in range(n):
        x = xl + i * dx
        y = line(x)
        ix = math.floor(x)
        iy = math.floor(y)
        fx = x - ix
        fy = y - iy
        res[0, i] = math.sqrt((x - xl) ** 2 + (y - yl) ** 2)
        res[1, i] = (1 - fx) * (1 - fy) * image[ix, iy] + fy * (1 - fx) * image[ix, iy + 1] + fx * (1 - fy) * image[ix + 1, iy] + fx * fy * image[ix + 1, iy + 1]
    return res


def aperture_curve(image, center, r_max, step, datatype='f4'):
    n = math.floor(r_max / step)
    res = numpy.ndarray(shape=(2, n), dtype=datatype)
    for i in range(n):
        r = i * step
        res[0, i] = r
        if r == 0:
            res[1, i] = 0
        else:
            ap = aperture.CircularAperture(center, r=r)
            res[1, i] = float(aperture.aperture_photometry(image, ap)["aperture_sum"][0])
    return res


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", action="store", default="data.fits", help="input FITS file")
args = parser.parse_args()
img = fits.open(args.input)
img_mean = mean_img(img)
img_median = median_img(img)

sl_mean = slicing(img_mean, Line(0, img_mean.shape[1] / 2), 0.1)
sl_median = slicing(img_median, Line(0, img_median.shape[1] / 2), 0.1)
plt.subplot(2, 1, 1)
plt.xlabel("x")
plt.ylabel("I")
plt.plot(sl_mean[0], sl_mean[1], color="#00FF00")
plt.subplot(2, 1, 2)
plt.xlabel("x")
plt.ylabel("I")
plt.plot(sl_median[0], sl_median[1], color="#0000FF")
fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.savefig("fig1.png", dpi=250)
plt.show()

curve_mean = aperture_curve(img_mean, (img_mean.shape[0] / 2, img_mean.shape[1] / 2), 100, 1)
curve_median = aperture_curve(img_median, (img_median.shape[0] / 2, img_median.shape[1] / 2), 100, 1)
plt.subplot(2, 1, 1)
plt.xlabel("r")
plt.ylabel("I")
plt.plot(curve_mean[0], curve_mean[1], color="#00FF00")
plt.subplot(2, 1, 2)
plt.xlabel("r")
plt.ylabel("I")
plt.plot(curve_median[0], curve_median[1], color="#0000FF")
fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.savefig("fig2.png", dpi=250)
plt.show()
