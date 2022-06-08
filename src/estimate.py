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
import matching as M


def compute_noise_curve_obsolete(img_raw_0, img_raw_1, bins=64, w=8, th=3, T=5, q=0.05, demosaic=True, search_range=3, num_div=16,
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
        channels = 4  # Only accept 4 channels so far

        img_raw_0 = np.array([img_raw_0[::2, ::2], img_raw_0[::2, 1::2],
                             img_raw_0[1::2, ::2], img_raw_0[1::2, 1::2]])
        img_raw_1 = np.array([img_raw_1[::2, ::2], img_raw_1[::2, 1::2],
                             img_raw_1[1::2, ::2], img_raw_1[1::2, 1::2]])
    else:
        if np.ndim(img_raw_0) == 2:
            # channels = 1
            img_raw_0 = img_raw_0[None, ...]
            img_raw_1 = img_raw_1[None, ...]
        else:  # == 3
            img_raw_0 = np.transpose(img_raw_0, (-1, 0, 1))
            img_raw_1 = np.transpose(img_raw_1, (-1, 0, 1))
        channels, H, W = img_raw_0.shape

    means_of_bins_all_channels = np.zeros((channels, bins))
    variances_of_bins_all_channels = np.zeros((channels, bins))

    for channel in range(channels):

        img_0_one_channel = img_raw_0[channel]
        img_1_one_channel = img_raw_1[channel]

        img_0_blocks, img_1_blocks = patch_match(
            img_0_one_channel, img_1_one_channel, w, th, search_range, num_div)

        # remove saturated blocks
        max_val = np.max([np.max(img_0_one_channel),
                         np.max(img_1_one_channel)])
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
        indices_of_means = indices_of_means[: len(
            indices_of_means) // bins * bins]

        means_sorted = means[indices_of_means]
        diff_blocks_sorted = diff_blocks[indices_of_means]

        diff_blocks_in_bins = diff_blocks_sorted.reshape(bins, -1, w, w)
        means_of_bins = means_sorted.reshape(bins, -1).mean(axis=-1)

        variances_of_bins = np.zeros((bins))

        # TODO: process all the bins in parallel
        for bin in range(bins):
            blocks_one_bin = diff_blocks_in_bins[bin]

            if auto_quantile:
                sigma = estimate_noise_iterative(blocks_one_bin, w, T)
            else:
                sigma = estimate_noise(blocks_one_bin, w, T, q)

            # the variance here contains double noise, should be divided by 2
            variances_of_bins[bin] = sigma * sigma / 2

        means_of_bins_all_channels[channel] = means_of_bins
        variances_of_bins_all_channels[channel] = variances_of_bins

    return means_of_bins_all_channels, variances_of_bins_all_channels


def estimate_noise_curve(img_ref, img_mov, w, T, th, q, bins, s, num_div=16, prec_level=2):
    """ Integrated pipeline: estimate noise curve from two successive images
        (See algo. 10 of sec. 5.6 in the paper)

    Parameters
    ----------
    img_ref: np.ndarray
        Reference image of size (C, H, W)
    img_mov: np.ndarray
        Moving image of size (C, H, W)
    w: int
        Block size
    T: int
        Threshold for separating the entries for low and high frequency DCT coefficents
    th: int
        Thickness of surrounding ring for matching
    q: float
        Percentile of blocks used for estimation
    bins: int
        number of bins
    s: int
        Half of search range for patch matching
        Note that the range of a squared search region window = search_range * 2 + 1
    num_div : int
        Number of divided areas of 360 degrees for gradient matching
    prec_level: int
        Subpixel precision = (1/2)^prec_level
        e.g. prec_level=3 leads to subpixel precision at 0.125 px
    Returns
    -------
    intensities: np.ndarray
        Intensities of the noise curve, of size (C, bins)
    variances: np.ndarray
        Variances of the noise curve, of size (C, bins)
    """

    assert img_ref.shape == img_mov.shape

    C, H, W = img_ref.shape

    intensities = np.zeros((C, bins))
    variances = np.zeros((C, bins))

    for ch in range(C):
        img_ref_chnl = img_ref[ch].astype(np.int32)
        img_mov_chnl = img_mov[ch].astype(np.int32)
        pos_ref, pos_mov = M.pixel_match(img_ref_chnl, img_mov_chnl, w, th, s, num_div)

        # print("M.pixel_match()")

        pos_ref_in_bins, pos_mov_in_bins = M.partition(img_ref_chnl, img_ref_chnl, pos_ref, pos_mov, w, bins)
        # print("M.partition()")
        pos_ref_filtered_in_bins = []
        pos_mov_filtered_in_bins = []
        for b in range(bins):
            pos_ref = pos_ref_in_bins[b]
            pos_mov = pos_mov_in_bins[b]

            pos_ref_filtered, pos_mov_filtered = M.filter_position_pairs(img_ref_chnl, img_ref_chnl, pos_ref, pos_mov, w, T, 3 * q)
            
            pos_ref_filtered_in_bins.append(pos_ref_filtered)
            pos_mov_filtered_in_bins.append(pos_mov_filtered)
        
        # Merge the pairs together so that they are processed at once in subpixel matching
        pos_ref_filtered_in_bins = np.vstack(pos_ref_filtered_in_bins)
        pos_mov_filtered_in_bins = np.vstack(pos_mov_filtered_in_bins)

        # print(pos_ref_filtered_in_bins.shape)
        # print(pos_mov_filtered_in_bins.shape)

        blks_ref_in_bins, blks_mov_in_bins = M.subpixel_match(img_ref_chnl, img_mov_chnl, pos_ref_filtered_in_bins, pos_mov_filtered_in_bins, \
                                              w, th, num_iter=prec_level)
        
        # print("blks_ref_in_bins", blks_ref_in_bins.shape)
        # print("blks_mov_in_bins", blks_mov_in_bins.shape)
        
        # blks_ref_filtered and blks_mov_filtered are already sorted by their intensities
        blks_ref_in_bins = blks_ref_in_bins[:int(len(blks_ref_in_bins) // bins * bins)].reshape(bins, -1, w, w)
        blks_mov_in_bins = blks_mov_in_bins[:int(len(blks_mov_in_bins) // bins * bins)].reshape(bins, -1, w, w)

        for b in range(bins):
            blks_ref = blks_ref_in_bins[b]
            blks_mov = blks_mov_in_bins[b]

            intensity = (np.mean(blks_ref) + np.mean(blks_mov)) / 2
            intensities[ch, b] = intensity

            # TODO: Add to IPOL paper
            # filter_block_pairs need real blocks as input, but not img + block positions

            blks_ref_filtered, blks_mov_filtered = M.filter_block_pairs(blks_ref, blks_mov, T, 1/3)
            variance = M.compute_variance_from_pairs(blks_ref_filtered, blks_mov_filtered, T)
            variances[ch, b] = variance

        print("intensities:", intensities)
        print("variances:", variances)

    return intensities, variances
