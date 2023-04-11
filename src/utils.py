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
import cv2
import matplotlib.pyplot as plt
import rawpy

# Set random state for reproducible results
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from skimage.util.shape import view_as_blocks
from scipy.ndimage import gaussian_filter
import skimage.io as iio


def read_img(fname: str, grayscale:bool=False):
    """ Read image from a file name, output a multi-channel image. Depending on the nature of the input image, the number of 
        channels is different:
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
        C = 3 for 
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
                with rawpy.imread(fname) as raw:
                    img = raw.raw_image.copy().astype(np.float32)
                success = True
            except:
                pass
    else:
        img = iio.imread(fname, plugin='pil').astype(np.float32)
    
    demosaic = False
    if fname.endswith(".dng") or fname.endswith(".tif") or fname.endswith(".tiff"):
        if not grayscale:
            demosaic = True

    # if fname.endswith(".tif") or fname.endswith(".tiff") or fname.endswith(".tif") or fname.endswith(".tif"):
    #     img = cv2.imread(fname, -1).astype(np.float32)
    #     if not grayscale:
    #         demosaic = True
    # elif fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg"):
    #     img = cv2.imread(fname, -1).astype(np.float32)

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

    RandomState(MT19937(SeedSequence(123456789)))

    noise = np.random.normal(0, np.sqrt(a + img_clean * b))
    img_noisy = img_clean + noise

    # IPOL input is in uint8, we support only the range of
    # [0, 255] for this moment
    img_noisy = np.clip(np.round(img_noisy), 0, 255).astype(np.int32)

    return img_noisy


def downscale(img: np.ndarray, antialias: bool=True) -> np.ndarray:
    """ Downscale an image by factor 2. Each of the 4 pixels in a 2x2 block
    is extracted to recompose a downsampled image. The 4 downsampled images
    are finally sliced with vertical and horizontal flipping.

    e.g. If input image is  
            A  B
            C  D
         then the output image looks like
            a b b a
            c d d c
            c d d c
            a b b a

    Parameters
    ----------
    img : np.ndarray
        Image of size (C, H, W)
    Returns
    -------
    img : np.ndarray
        Image of size (C, H, W)
    """
    
    gray_img = False
    if np.ndim(img) == 2:
        img = img[None, ...]
        gray_img = True

    C, H, W = img.shape

    img_ds = np.zeros_like(img)

    if antialias:
        antialiasing = 1.0
        sigma = antialiasing * np.sqrt(2**2 - 1)
        img = gaussian_filter(img, sigma=sigma, mode='reflect')

    for c in range(C):
        img_ch = img[c]
        img_top_l = img_ch[::2, ::2]

        img_top_r = img_ch[::2, 1::2]
        img_top_r = np.flip(img_top_r, 1)

        img_bot_l = img_ch[1::2, ::2]
        img_bot_l = np.flip(img_bot_l, 0)

        img_bot_r = img_ch[1::2, 1::2]
        img_bot_r = np.flip(img_bot_r, (0, 1))

        # img_top = np.concatenate((img_top_l, img_top_r), axis=1)
        # img_bot = np.concatenate((img_top_l, img_top_r), axis=1)

        img_ch = np.block([[img_top_l, img_top_r], [img_bot_l, img_bot_r]])

        img_ds[c] = img_ch

    if gray_img:
        img_ds = img_ds[0]

    return img_ds



def downscale_lebrun(img: np.ndarray) -> np.ndarray:
    """ Downscale an image by factor 2. See multiscale blind denoising.

    e.g. If input image is  
            A  B
            C  D
         then the output image looks like
            a b b a
            c d d c
            c d d c
            a b b a

    Parameters
    ----------
    img : np.ndarray
        Image of size (C, H, W)
    Returns
    -------
    img : np.ndarray
        Image of size (C, H, W)
    """
    
    gray_img = False
    if np.ndim(img) == 2:
        img = img[None, ...]
        gray_img = True

    C, H, W = img.shape

    # Crop the image into odd size
    if H % 2 == 0: H = H - 1
    if W % 2 == 0: W = W - 1
    img = img[:, :H, :W]

    img_ds = np.zeros((C, H-1, W-1), dtype=img.dtype)


    for c in range(C):
        img_ch = img[c]
        img_top_l = img_ch[:H//2*2, :W//2*2]
        img_top_l = view_as_blocks(img_top_l, (2,2)).mean(axis=(-1, -2))

        img_top_r = img_ch[:H//2*2, 1:(W-1)//2*2+1]
        img_top_r = view_as_blocks(img_top_r, (2,2)).mean(axis=(-1, -2))
        img_top_r = np.flip(img_top_r, 1)

        img_bot_l = img_ch[1:(H-1)//2*2+1, :W//2*2]
        img_bot_l = view_as_blocks(img_bot_l, (2,2)).mean(axis=(-1, -2))
        img_bot_l = np.flip(img_bot_l, 0)

        img_bot_r = img_ch[1:(H-1)//2*2+1, 1:(W-1)//2*2+1]
        img_bot_r = view_as_blocks(img_bot_r, (2,2)).mean(axis=(-1, -2))
        img_bot_r = np.flip(img_bot_r, (0, 1))

        img_ch = np.block([[img_top_l, img_top_r], [img_bot_l, img_bot_r]])

        img_ds[c] = img_ch

    if gray_img:
        img_ds = img_ds[0]

    return img_ds



def downscale_once(img: np.ndarray) -> np.ndarray:
    """ Downscale an image by factor 2. See multiscale blind denoising.

    e.g. If input image is  
            A  B
            C  D
         then the output image looks like
            a b b a
            c d d c
            c d d c
            a b b a

    Parameters
    ----------
    img : np.ndarray
        Image of size (C, H, W)
    Returns
    -------
    img : np.ndarray
        Image of size (C, H, W)
    """
    
    gray_img = False
    if np.ndim(img) == 2:
        img = img[None, ...]
        gray_img = True

    C, H, W = img.shape

    # Crop the image into even size
    if H % 2 != 0: H = H - 1
    if W % 2 != 0: W = W - 1
    img = img[:, :H, :W]

    img_ds = np.zeros((C, H//2, W//2), dtype=img.dtype)

    for c in range(C):
        img_ch = img[c]
        img_ch = img_ch[:H//2*2, :W//2*2]
        img_ch = view_as_blocks(img_ch, (2,2)).mean(axis=(-1, -2))
        img_ds[c] = img_ch

    if gray_img:
        img_ds = img_ds[0]

    return img_ds


def multi_downscale(img: np.ndarray, max_scale=0) -> np.ndarray:
    """ Apply downscale for multiple times, and obtain multi-scale images.

    Parameters
    ----------
    img : np.ndarray
        Image of size (C, H, W)
    Returns
    -------
    img : np.ndarray
        Image of size (N, C, H, W).
        N is the number of scales.
    """

    s = 0

    img_mds = []
    img_mds.append(img)

    while s < max_scale:
        img = downscale(img)
        img_mds.append(img)
        s += 1

    img_mds = np.array(img_mds)

    return img_mds


def plot_noise_curve(intensities, variances, a=None, b=None, fname=None):
    """ Plot the noise curve {means <=> variances}
        and save it to an image file

    Parameters
    ----------
    intensities : numpy
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
    # plt.title('Noise curves from frame discrepancy', fontsize=20)
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
