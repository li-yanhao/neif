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


def read_img(fname):
    """ Read image from a file name

    Parameters
    ----------
    fname : str
        Filename of the image
    
    Returns
    ----------
    img : np.ndarray
        Image in int32, of size (H, W, C) for multi-channel image
        or (H, W) for single channel image
    """

    if fname.endswith(".tiff"):
        with rawpy.imread(fname) as raw:
            img = raw.raw_image.copy().astype(np.int32)
    elif fname.endswith(".png"):
        img = cv2.imread(fname, -1).astype(np.int32)

    return img


def plot_noise_curve(means, variances, fname="noise_curve.png"):
    """ Plot the noise curve {means <=> variances}
        and save it to an image file

    Parameters
    ----------
    means : numpy
        Means of bins for multiple channels, in size (channels, bins)
    variances : numpy.ndarray
        Variances of bins for multiple channels, in size (channels, bins)
    confidences : numpy.ndarray
        TBC
    fname : str
        Saved image file name

    Returns
    -------
    """

    assert means.shape == variances.shape, "\'means\' and \'variances\' should have same size"
    channels, bins = means.shape
    

    # global lock_plt
    # lock_plt.acquire()
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(800*px, 600*px))
    plt.clf()
    plt.title('Noise curves from inter-frame discrepancy', fontsize=20)
    for channel in range(channels):
        plt.plot(means[channel], variances[channel], '-o', label='var of channel ' + str(channel))
    plt.legend()
    plt.xlabel('intensity', fontsize=15)
    plt.ylabel('variance', fontsize=15)
    plt.grid()
    
    plt.savefig(fname)
    plt.close()
    # lock_plt.release()