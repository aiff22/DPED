import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf


def _FSpecialGauss(size, sigma):

    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1

    if size % 2 == 0:
        offset = 0.5
        stop -= 1

    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))

    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:

        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')

    else:

        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2

    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)

    return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):

    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size

    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]

    mssim = np.array([])
    mcs = np.array([])

    for _ in range(levels):

        ssim, cs = _SSIMForMultiScale(im1, im2, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)

        filtered = [convolve(im, downsample_filter, mode='reflect') for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]

    return np.prod(mcs[0:levels-1] ** weights[0:levels-1]) * (mssim[levels-1] ** weights[levels-1])
