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
import matching as M
import time 

from skimage.util import view_as_windows, view_as_blocks


def estimate_noise_curve(img_ref, img_mov, w: int, T: int, th: int, q: float, bins: int, s: int, scale=0):
    """ Main function: estimate noise curves from two successive images
        (See Algo. 7 of Sec. 5 and Algo. 9 of Sec. 8 in the paper)

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
        Percentile of blocks used for estimation, in [0, 1]
    bins: int
        number of bins
    s: int
        Half of search range for patch matching
        Note that the range of a squared search region window = search_range * 2 + 1
    scale: int
        The downsampling scale. The image block pairs will be downsampled by factor 2^scale. 0 for no downscaling, 

    Returns
    -------
    intensities: np.ndarray
        Intensities of the noise curve, of size (C, bins)
    variances: np.ndarray
        Variances of the noise curve, of size (C, bins)
    """

    assert img_ref.shape == img_mov.shape

    img_ref = img_ref.astype(np.float32)

    C, H, W = img_ref.shape

    intensities = np.zeros((C, bins))
    variances = np.zeros((C, bins))

    # Use larger block for matching, so that difference blocks can be subsampled at correct size
    f = 2**scale # downscaling factor
    w_up = f * w

    for ch in range(C):
        img_ref_chnl = img_ref[ch]
        img_mov_chnl = img_mov[ch]
        pos_ref, pos_mov = M.pixel_match(
            img_ref_chnl, img_mov_chnl, w_up, th, s)

        pos_ref, pos_mov = M.remove_saturated(
            img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w_up)

        pos_ref_in_bins, pos_mov_in_bins = M.partition(
            img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w_up, bins)

        blks_ref = view_as_windows(img_ref_chnl, (w_up, w_up), step=(1, 1))
        blks_mov = view_as_windows(img_mov_chnl, (w_up, w_up), step=(1, 1))

        for b in range(bins):
            pos_ref = pos_ref_in_bins[b]
            pos_mov = pos_mov_in_bins[b]

            pos_ref_selected, pos_mov_selected = M.select_position_pairs(
                img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w_up, T, q)

            blks_ref_in_bin = blks_ref[pos_ref_selected[:,0], 
                                       pos_ref_selected[:,1]].astype(np.float32)

            blks_mov_in_bin = blks_mov[pos_mov_selected[:,0], 
                                       pos_mov_selected[:,1]].astype(np.float32)

            variance = M.compute_variance_from_pairs(
                blks_ref_in_bin, blks_mov_in_bin, T, f)
            intensity = (blks_ref_in_bin.mean() + blks_mov_in_bin.mean()) / 2

            variances[ch, b] = variance
            intensities[ch, b] = intensity

    return intensities, variances


def estimate_noise_curve_subpixel(img_ref, img_mov, w: int, T: int, th: int, q: float, bins: int, s: int, subpx_order: int = 0, scale=0):
    """ Main function: estimate noise curves from two successive images

    Parameters
    ----------
    img_ref: np.ndarray
        Reference image of size (C, H, W).
    img_mov: np.ndarray
        Moving image of size (C, H, W).
    w: int
        Block size.
    T: int
        Threshold for separating the entries for low and high frequency DCT coefficents.
    th: int
        Thickness of surrounding ring for matching.
    q: float
        Percentile of blocks used for estimation.
    bins: int
        Number of bins.
    s: int
        Half of search range for patch matching.
        Note that the range of a squared search region window = search_range * 2 + 1.
    subpx_order: int
        The scale of upsampling for subpixel matching. 
        The upsampling factor is 2^subpx_order and the subpixel matching precision = (1/2)^subpx_order.
        0 for no subpixel matching.
    scale: int
        The scale of downscaling. The downscaling factor is 2^scale. 0 for no downscaling.

    Returns
    -------
    intensities: np.ndarray
        Intensities of the noise curve, of size (C, bins).
    variances: np.ndarray
        Variances of the noise curve, of size (C, bins).
    """

    assert img_ref.shape == img_mov.shape

    C, H, W = img_ref.shape

    intensities = np.zeros((C, bins))
    variances = np.zeros((C, bins))

    # Use larger block for matching, so that difference blocks can be subsampled at correct size
    w_match = 2**scale * w


    for ch in range(C):
        img_ref_chnl = img_ref[ch]
        img_mov_chnl = img_mov[ch]
        print("start pixel_match", time.time())
        pos_ref, pos_mov = M.pixel_match(img_ref_chnl, img_mov_chnl, w_match, th, s)
        print("end pixel_match", time.time())

        print("start M.remove_saturated", time.time())
        pos_ref, pos_mov = M.remove_saturated(
            img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w_match)
        print("end M.remove_saturated", time.time())

        # remove blocks near the borders so that subpixel matching can work
        border = 4
        mask = (pos_mov[:, 0] >= border + th) & (pos_mov[:, 0] < H - w_match + 1 - th - border) \
            & (pos_mov[:, 1] >= border + th) & (pos_mov[:, 1] < W - w_match + 1 - th - border)
        pos_ref = pos_ref[mask]
        pos_mov = pos_mov[mask]

        print("start M.partition", time.time())
        pos_ref_in_bins, pos_mov_in_bins = M.partition(
            img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w_match, bins)
        print("end M.partition", time.time())

        pos_ref_selected_in_bins = []
        pos_mov_selected_in_bins = []

        for b in range(bins):
            pos_ref = pos_ref_in_bins[b]
            pos_mov = pos_mov_in_bins[b]

            pos_ref_selected, pos_mov_selected = M.select_position_pairs(
                img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w_match, T, 3 * q)

            pos_ref_selected_in_bins.append(pos_ref_selected)
            pos_mov_selected_in_bins.append(pos_mov_selected)

        pos_ref_selected_in_bins = np.vstack(pos_ref_selected_in_bins)
        pos_mov_selected_in_bins = np.vstack(pos_mov_selected_in_bins)

        print("start subpixel_match", time.time())
        if subpx_order > 0:
            blks_ref_in_bins, blks_mov_in_bins = M.subpixel_match(
                img_ref_chnl, img_mov_chnl, pos_ref_selected_in_bins, pos_mov_selected_in_bins,
                w_match, th, order=subpx_order) # (N, w, w)
        elif subpx_order == 0:
            blks_ref = view_as_windows(img_ref_chnl, (w_match, w_match), step=(1, 1))
            blks_mov = view_as_windows(img_mov_chnl, (w_match, w_match), step=(1, 1))
            blks_ref_in_bins = blks_ref[pos_ref_selected_in_bins[:, 0], 
                                        pos_ref_selected_in_bins[:, 1]]
            blks_mov_in_bins = blks_mov[pos_mov_selected_in_bins[:, 0],
                                        pos_mov_selected_in_bins[:, 1]]
            blks_ref_in_bins = blks_ref_in_bins.astype(np.float32)
            blks_mov_in_bins = blks_mov_in_bins.astype(np.float32)
        else:
            raise Exception("`scale` should be 0 or positive")
        print("end subpixel_match", time.time())

        # downsample the matched blocks to the required scale with average filter
        print("start avg_filter", time.time())
        blks_ref_in_bins = np.ascontiguousarray(blks_ref_in_bins)
        blks_mov_in_bins = np.ascontiguousarray(blks_mov_in_bins)
        blks_ref_in_bins = np.mean(view_as_blocks(blks_ref_in_bins, (1, 2**scale, 2**scale)), axis=(-1, -2, -3)) # (N, w, w)
        blks_mov_in_bins = np.mean(view_as_blocks(blks_mov_in_bins, (1, 2**scale, 2**scale)), axis=(-1, -2, -3)) # (N, w, w)
        print("end avg_filter", time.time())

        # blks_ref_selected and blks_mov_selected are already sorted by their intensities
        blks_ref_in_bins = blks_ref_in_bins[:int(
            len(blks_ref_in_bins) // bins * bins)].reshape(bins, -1, w, w)
        blks_mov_in_bins = blks_mov_in_bins[:int(
            len(blks_mov_in_bins) // bins * bins)].reshape(bins, -1, w, w)

        for b in range(bins):
            blks_ref = blks_ref_in_bins[b]
            blks_mov = blks_mov_in_bins[b]

            intensity, variance = M.estimate_intensity_and_variance(
                blks_mov, blks_ref, T, 1/3)

            variances[ch, b] = variance
            intensities[ch, b] = intensity

    return intensities, variances


def compute_median_curve(in_curves):
    """ Compute the median curve given a set of curves
        (See Algo. 8 of Sec. 5 in the paper)

    Parameters
    ----------
    in_curves : numpy.ndarray
        A set of n curves, each curve contains k points (xi, yi), 
        with xi in the ascending order in each curve.
        Of size (n, 2, k).

    Returns
    -------
    numpy.ndarray
        Median curve from the curves, of size (2, k)
    """
    assert in_curves.ndim == 3 and in_curves.shape[1] == 2
    n, _, k = in_curves.shape
    x_ctrls = np.median(in_curves[:, 0, :], axis=0)

    y_ctrls_all = np.zeros((n, k))
    y_ctrls_median = np.zeros(k)
    for i in range(n):
        curve = in_curves[i]
        x = curve[0]
        y = curve[1]
        y_ctrls = np.interp(x_ctrls, x, y, left=np.nan, right=np.nan)
        y_ctrls_all[i] = y_ctrls

    for i in range(k):
        y = y_ctrls_all[:, i]
        y_ctrls_median[i] = np.median(y[~np.isnan(y)])

    return np.vstack([x_ctrls, y_ctrls_median])
