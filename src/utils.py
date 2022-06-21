# This file is part of the algorithm 
# "Video Signal-Dependent Noise Estimation via Inter-Frame Prediction"

# Copyright(c) 2022 Yanhao Li.
# yanhao.li@outlook.com

# This file may be licensed under the terms of of the
# GNU General Public License Version 2 (the ``GPL'').

# Software distributed under the License is distributed
# on an ``AS IS'' basis, WITHOUT WARRANTY OF ANY KIND, either
# express or implied. See the GPL for the specific language
# governing rights and limitations.

# You should have received a copy of the GPL along with this
# program. If not, go to http://www.gnu.org/licenses/gpl.html
# or write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import rawpy

# Set random state for reproducible results
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


def read_img(fname:str, demosaic:bool):
    """ Read image from a file name

    Parameters
    ----------
    fname: str
        Filename of the image
    demosaic: bool
        Whether process demosaicking or not

    Returns
    -------
    img : np.ndarray
        Image in int32, of size (C, H, W)
        C = 1 for grayscale image
    """

    if fname.endswith(".tiff") or fname.endswith(".dng"):
        with rawpy.imread(fname) as raw:
            img = raw.raw_image.copy().astype(np.int32)
    elif fname.endswith(".png"):
        img = cv2.imread(fname, -1).astype(np.int32)


    if demosaic:
        if np.ndim(img) == 3:
            raise Exception("Cannot demosaick a multi-channel image")
        # H, W = img.shape
        # Only accept 4 channels so far
        img = np.array([img[::2, ::2], img[::2, 1::2], img[1::2, ::2], img[1::2, 1::2]])
    else:
        if np.ndim(img) == 2:
            # channels = 1, grayscale image
            img = img[None, ...]
        else: 
            # channels == 3, color image
            img = np.transpose(img, (2, 0, 1))
    
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
    plt.title('Noise curves from frame discrepancy', fontsize=20)
    for channel in range(channels):
        plt.plot(intensities[channel], variances[channel], '-o', label='var of channel ' + str(channel))
    
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