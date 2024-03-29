# This file is part of the algorithm
# "A Signal-Dependent Video Noise Estimator via Inter-frame Signal Suppression"


# Copyright (c) 2022 Yanhao Li
# yanhao.li@outlook.com

# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt
import rawpy

import skimage.io as iio


def read_img(fname: str, grayscale: bool = False) -> np.ndarray:
    """ Read image from a file name, output a multi-channel image. Depending on the nature of the input image, the number of output channels is different:
        - raw image in grayscale: number of channels = 1
        - raw image in color: number of channels = 4
        - processed image in grayscale: number of channels = 1
        - processed image in color: number of channels = 3
        The input file will be recognized as raw image if the suffix is .tif, .tiff or .dng, and as processed image if the suffix is .png, .jpg or .jpeg.

    Parameters
    ----------
    fname: str
        Filename of the image
    demosaic: bool
        Whether process demosaicking or not

    Returns
    -------
    img : np.ndarray
        Image in float32, of size (C, H, W)
        C = 1 for grayscale image
        C = 3 for color image
    """

    if fname.endswith(".dng"):
        with rawpy.imread(fname) as raw:
            img = raw.raw_image.copy().astype(np.float32)
    elif fname.endswith(".tif") or fname.endswith(".tiff"):
        success = False
        try:
            with rawpy.imread(fname) as raw:
                img = raw.raw_image.copy().astype(np.float32)
            success = True
        except:
            pass
        if not success:
            try:
                img = iio.imread(fname, plugin='pil').astype(np.float32)
                success = True
            except:
                pass
        if not success:
            raise Exception("Failed to read `{}`.".format(fname))
    elif fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg"):
        img = iio.imread(fname, plugin='pil')
        if np.ndim(img) == 2:
            img = img[..., None]
    else:
        raise NotImplementedError(
            "Image filename `{}` is invalid. Only `.tif`, `.tiff`, `.dng`, `.png`, `.jpg` and `.jpeg` formats are support. ".format(fname))

    demosaic = False
    if fname.endswith(".dng") or fname.endswith(".tif") or fname.endswith(".tiff"):
        if not grayscale:
            demosaic = True

    if demosaic:
        # Only accept 4 channels so far
        img = np.array([img[::2, ::2], img[::2, 1::2],
                       img[1::2, ::2], img[1::2, 1::2]])
    else:
        if np.ndim(img) == 2:
            # channels = 1, grayscale image
            img = img[None, ...]
        else:
            # channels == 3, color image
            img = np.transpose(img, (2, 0, 1))

    img = np.ascontiguousarray(img)
    assert np.ndim(img) == 3

    return img


def save_img(prefix: str, img: np.ndarray) -> None:
    """ Save an image for visualizing
    Parameters
    ----------
    fname: str
        Filename of the image to save
    img: np.ndarray
        An image of size (C, H, W)
    """
    assert len(img.shape) == 3
    C, H, W = img.shape
    img = np.transpose(img, (1, 2, 0))

    if C == 4:
        # we assume the bayer pattern is BGGR, and take only one G channel
        img_rgb = np.zeros((H, W, 3))
        img_rgb[:, :, 0] = img[:, :, 3]
        img_rgb[:, :, 1] = img[:, :, 1]
        img_rgb[:, :, 2] = img[:, :, 0]
        img = img_rgb
        if np.max(img) > 255:
            img = img / np.max(img) * 255
        iio.imsave(prefix + ".png", img.astype(np.uint8))
        return

    if C == 1:
        img = img[:, :, 0]
    iio.imsave(prefix + ".png", img.astype(np.uint8))


def save_noise(prefix: str, noise: np.ndarray) -> None:
    """ Save an image of noise residual for visualizing

    Parameters
    ----------
    prefix: str
        The filename without extension for saving
    noise: np.ndarray
        An image of noise residual of size (C, H, W)
    """

    assert len(noise.shape) == 3
    C, H, W = noise.shape
    noise = np.transpose(noise, (1, 2, 0))

    img = np.abs(noise)
    if C == 4:
        if np.max(img) > 255:
            img = img / np.max(img) * 255
        iio.imsave(prefix + ".png", img.astype(np.uint8))
        return

    if C == 1:
        img = img[:, :, 0]
    iio.imsave(prefix + ".png", img.astype(np.uint8))


def add_noise(img_clean, a, b):
    """ Add simulated noise to a clean image

    Parameters
    ----------
    img_clean: np.ndarray
        Noiseless image
    a: float
        Param of noise model V = a + b * I
    b: float
        Param of noise model V = a + b * I

    Returns
    -------
    img_noisy: np.ndarray
        Image with added noise
    """

    noise = np.random.normal(0, np.sqrt(a + img_clean * b))
    img_noisy = img_clean + noise

    # IPOL input is in uint8, we support only the range of
    # [0, 255] for this moment
    img_noisy = np.clip(np.round(img_noisy), 0, 255).astype(np.int32)

    return img_noisy, noise


def plot_noise_curve(intensities:np.ndarray, variances:np.ndarray, a:float=None, b:float=None, fname:str=None) -> None:
    """ Plot the noise curve {means <=> variances}
        and save it to an image file

    Parameters
    ----------
    intensities : np.ndarray
        Means of bins for multiple channels, in size (channels, bins)
    variances : numpy.ndarray
        Variances of bins for multiple channels, in size (channels, bins)
    a: float
        Param of noise model V = a + b * I
    b: float
        Param of noise model V = a + b * I
    fname : str
        Saved image file name

    Returns
    -------
    """
    assert intensities.shape == variances.shape, "\'intensities\' and \'variances\' should have same size"
    channels, bins = intensities.shape

    global lock_plt
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(800*px, 600*px))
    plt.clf()
    for channel in range(channels):
        plt.plot(intensities[channel], variances[channel],
                 '-o', label='var of channel ' + str(channel))

    if a is not None and b is not None:
        xs = np.linspace(0, np.max(intensities), 50)
        ys = a + b * xs
        plt.plot(xs, ys, '--r', label='ground truth ')

    plt.legend()
    plt.xlabel('intensity', fontsize=15)
    plt.ylabel('variance', fontsize=15)
    plt.grid()

    if fname is None:
        fname = "noise_curve.png"
    plt.savefig(fname)
    plt.close()
