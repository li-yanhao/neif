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
from scipy.fft import dctn
from scipy.stats import chi2
from .patch_match import patch_match


def low_pass_matrix(w=8, T=5):
    """ Give a low-pass matrix M composed of 0 and 1
        M[i,j] = 1      if i + j <= T and i + j != 0
        M[i,j] = 0      else

    Parameters
    ----------
    w : int
        Size of the squared low-pass matrix
    T : int
        Threshold for entries associated to 1

    Returns
    ----------
    numpy.ndarray
        The low-pass matrix, in size (w, w)
    """
    M = np.zeros((w, w)).astype(np.int8)
    for i in range(w):
        for j in range(w):
            # M[i, j] = 1 if (i+j<=T and i+j != 0) else 0
            M[i, j] = 1 if (i+j<=T) else 0
    
    return M


def estimate_noise_iterative(blocks, w, T, num_iter=15):
    """ Estimate the noise from a set of blocks in an iterative way.
        In each iteration the blocks of pure noise will be selected
        and the noise variance will be updated

    Parameters
    ----------
    blocks : numpy.ndarray
        Blocks of pixel values, in size (n, w, w)
    w : int
        Block size
    T : int
        Threshold for entries associated to low frequencies,
        coefficient at (i,j) when i+j<=T and i+j!=0 is considered as low frequency coefficient
    num_iter : int
        Number of iterations

    Returns
    ----------
    float64
        Estimated noise
    """

    assert len(blocks.shape) == 3
    assert blocks.shape[1] == blocks.shape[2] == w
    
    
    delta_matrix = low_pass_matrix(w, T)
    

    blocks_dct = dctn(blocks, axes=(-1,-2), norm='ortho', workers=15)
    # print(blocks)

    low_freq_energies = (blocks_dct * blocks_dct * delta_matrix).sum(axis=(-1, -2)) / delta_matrix.sum()
    # print(low_freq_energies)

    indices_of_energies = np.argsort(low_freq_energies)
    block_dct_sorted_by_energy = blocks_dct[indices_of_energies]
    # flat_blocks_dct = blocks_dct[indices_of_energies[: int(q * len(blocks_dct))]]


    high_freq_delta_matrix = ~delta_matrix.astype(bool)
    high_freq_delta_matrix[0, 0] = False

    k = np.sum(high_freq_delta_matrix)
    high_freq_coefs = block_dct_sorted_by_energy[:, high_freq_delta_matrix] # N, k

    # TODO: Drop blocks with non-normal high frequency coefficients
    variance = np.median((high_freq_coefs * high_freq_coefs).mean(axis=0))
    
    SQ_hf = np.sum(high_freq_coefs * high_freq_coefs, axis=1) # N
    tau = chi2.ppf(0.90, k, loc=0, scale=1)

    # num_iter = 30
    for _ in range(num_iter):

        valid_blocks_mask = SQ_hf < (tau * variance)
        valid_high_freq_coefs = high_freq_coefs[valid_blocks_mask]
        variance = np.median((valid_high_freq_coefs * valid_high_freq_coefs).mean(axis=0))
        # print(f"variance: {variance:.2f}")
    # print("estimate noise done")
            
    return np.sqrt(variance)


def estimate_noise(blocks, w, T, q):
    """ Estimate the noise from a set of blocks of pixels
        (Poisson or Gaussian model?)

    Parameters
    ----------
    blocks : numpy.ndarray
        Blocks of pixel values, in size (n, w, w)
    w : int
        Block size
    T : int
        Threshold for entries associated to low frequencies,
        coefficient at (i,j) when i+j<=T and i+j!=0 is considered as low frequency coefficient
    q : float
        Quantile of blocks with lowest low-frequency energies

    Returns
    ----------
    float
        Estimated noise
    """

    assert len(blocks.shape) == 3
    assert blocks.shape[1] == blocks.shape[2] == w
    
    delta_matrix = low_pass_matrix(w, T) # TODO: set it as parameter
    

    blocks_dct = dctn(blocks, axes=(-1,-2), norm='ortho', workers=15)

    low_freq_energies = (blocks_dct * blocks_dct * delta_matrix).sum(axis=(-1, -2)) / delta_matrix.sum()

    indices_of_energies = np.argsort(low_freq_energies)
    block_dct_sorted_by_energy = blocks_dct[indices_of_energies]
    # flat_blocks_dct = blocks_dct[indices_of_energies[: int(q * len(blocks_dct))]]


    # TODO: Drop blocks with non-normal high frequency coefficients
    high_freq_delta_matrix = ~delta_matrix.astype(bool)
    high_freq_delta_matrix[0, 0] = False

    high_freq_coefs = block_dct_sorted_by_energy[:, high_freq_delta_matrix]

    nb_blocks_max = int(q * len(blocks_dct))

    high_freq_coefs = high_freq_coefs[:nb_blocks_max]
    sigma = np.sqrt(np.median((high_freq_coefs * high_freq_coefs).mean(axis=0)))
            
    return sigma




def compute_noise_curve(img_raw_0, img_raw_1, bins=64, w=8, th=3, T=5, q=0.05, demosaic=True, search_range=3, num_div=16,
                        auto_quantile=False):
    """ Calculate the noise curve from a difference image

    Parameters
    ----------
    img_raw_0 : numpy.ndarray
        Raw image at t, in size (H, W)
    img_raw_1 : numpy.ndarray
        Raw image at t+1, in size (H, W)
    bins : int
        Number of bins for partition
    w : int
        Block size
    th : int
        Thickness of bounding ring of each patch for matching
    T : int
        Threshold for entries associated to low frequencies,
        coefficient at (i,j) when i+j<=T and i+j!=0 is considered as low frequency coefficient
    q : float
        Quantile of blocks with lowest low-frequency energies
    demosaic : bool
        A flag indicating if image needs demosaicing
    search_range : int
        Half of search range for patch matching. Note that the range of a squared search region window = search_range * 2 + 1
    num_div : int
        Number of divided areas of 360 degrees for gradient matching
    auto_quantile : bool
        Whether use fixed quantile of blocks or automatically choose the blocks with pure noise for estimation

    Returns
    ----------
    (numpy.ndarray, numpy.ndarray)
        [0]: Means of bins for 4 channels, in size (4, bins)
        [1]: Variances (noise) of bins for 4 channels, in size (4, bins)
    """
    
    assert img_raw_0.shape == img_raw_1.shape
    
    if demosaic:
        H, W = img_raw_0.shape
        channels = 4 # Only accept 4 channels so far

        img_raw_0 = np.array([img_raw_0[::2, ::2], img_raw_0[::2, 1::2], img_raw_0[1::2, ::2], img_raw_0[1::2, 1::2]])
        img_raw_1 = np.array([img_raw_1[::2, ::2], img_raw_1[::2, 1::2], img_raw_1[1::2, ::2], img_raw_1[1::2, 1::2]])
    else:
        if np.ndim(img_raw_0) == 2:
            # channels = 1
            img_raw_0 = img_raw_0[None, ...]
            img_raw_1 = img_raw_1[None, ...]
        else: # == 3
            img_raw_0 = np.transpose(img_raw_0, (-1, 0, 1))
            img_raw_1 = np.transpose(img_raw_1, (-1, 0, 1))
        channels, H, W = img_raw_0.shape
    
    
    means_of_bins_all_channels = np.zeros((channels, bins))
    variances_of_bins_all_channels = np.zeros((channels, bins))

    for channel in range(channels):
        
        img_0_one_channel = img_raw_0[channel]
        img_1_one_channel = img_raw_1[channel]

        
        img_0_blocks, img_1_blocks = patch_match(img_0_one_channel, img_1_one_channel, w, th, search_range, num_div)
        
        # remove saturated blocks
        max_val = np.max([np.max(img_0_one_channel), np.max(img_1_one_channel)])
        non_saturated_bitmap_0 = np.all(img_0_blocks < max_val, axis=(-2, -1))
        non_saturated_bitmap_1 = np.all(img_1_blocks < max_val, axis=(-2, -1))
        non_saturated_bitmap = non_saturated_bitmap_0 & non_saturated_bitmap_1

        img_0_blocks = img_0_blocks[non_saturated_bitmap]
        img_1_blocks = img_1_blocks[non_saturated_bitmap]

        # diff_blocks = img_0_blocks - img_1_blocks

        means = np.mean((img_0_blocks + img_1_blocks) / 2, axis=(-2, -1))

        diff_blocks = (img_0_blocks - img_1_blocks).reshape(-1, w, w)

        indices_of_means = np.argsort(means)

        
        # drop means with too high values
        indices_of_means = indices_of_means[: len(indices_of_means) // bins * bins]

        means_sorted = means[indices_of_means]
        diff_blocks_sorted = diff_blocks[indices_of_means]


        diff_blocks_in_bins = diff_blocks_sorted.reshape(bins, -1, w, w)
        means_of_bins = means_sorted.reshape(bins, -1).mean(axis=-1)

        variances_of_bins = np.zeros((bins))
        

        #TODO: process all the bins in parallel
        for bin in range(bins):
            blocks_one_bin = diff_blocks_in_bins[bin]

            if auto_quantile:
                sigma = estimate_noise_iterative(blocks_one_bin, w, T)
            else:
                sigma = estimate_noise(blocks_one_bin, w, T, q)

            variances_of_bins[bin] = sigma * sigma / 2 # the variance here contains double noise, should be divided by 2
            

        means_of_bins_all_channels[channel] = means_of_bins
        variances_of_bins_all_channels[channel] = variances_of_bins

    return means_of_bins_all_channels, variances_of_bins_all_channels
