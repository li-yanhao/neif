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

from skimage.util import view_as_windows


def estimate_noise_curve(img_ref, img_mov, w: int, T: int, th: int, q: float, bins: int, s: int, f=1):
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
        Percentile of blocks used for estimation
    bins: int
        number of bins
    s: int
        Half of search range for patch matching
        Note that the range of a squared search region window = search_range * 2 + 1
    f: int
        The factor of subscaling, 1 for no subscaling

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

    # Use larger block for matching, so that difference blocks can be subsampled at correct size
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

            blks_ref_in_bin = blks_ref[pos_ref_selected[:,
                                                        0], pos_ref_selected[:, 1]].astype(np.float32)

            blks_mov_in_bin = blks_mov[pos_mov_selected[:,
                                                        0], pos_mov_selected[:, 1]].astype(np.float32)

            variance = M.compute_variance_from_pairs(
                blks_ref_in_bin, blks_mov_in_bin, T, f)
            intensity = (blks_ref_in_bin.mean() + blks_mov_in_bin.mean()) / 2

            variances[ch, b] = variance
            intensities[ch, b] = intensity

    return intensities, variances


def estimate_noise_curve_subpixel(img_ref, img_mov, w: int, T: int, th: int, q: float, bins: int, s: int, prec_lvl: int = 0, scale=0):
    """ Main function: estimate noise curves from two successive images

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
    prec_lvl: int
        Subpixel precision = (1/2)^prec_lvl
        e.g. prec_lvl=3 leads to subpixel precision at 0.125 px
    scale: int
        The scale of subscaling, 0 for no subscaling

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

    # Use larger block for matching, so that difference blocks can be subsampled at correct size
    w = 2**scale * w

    for ch in range(C):
        img_ref_chnl = img_ref[ch]
        img_mov_chnl = img_mov[ch]
        pos_ref, pos_mov = M.pixel_match(img_ref_chnl, img_mov_chnl, w, th, s)

        pos_ref, pos_mov = M.remove_saturated(
            img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w)

        # remove blocks near the borders so that subpixel matching can work
        border = 2
        mask = (pos_mov[:, 0] >= border + th) & (pos_mov[:, 0] < H - w + 1 - th - border) \
            & (pos_mov[:, 1] >= border + th) & (pos_mov[:, 1] < W - w + 1 - th - border)
        pos_ref = pos_ref[mask]
        pos_mov = pos_mov[mask]

        # return pos_ref, pos_mov

        # blks_ref = view_as_windows(img_ref_chnl, (w, w), step=(1, 1))[pos_ref[:, 0], pos_ref[:, 1]]
        # blks_mov = view_as_windows(img_mov_chnl, (w, w), step=(1, 1))[pos_mov[:, 0], pos_mov[:, 1]]
        # return pos_ref, pos_mov

        # intensities = M.partition(img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w, bins)

        pos_ref_in_bins, pos_mov_in_bins = M.partition(
            img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w, bins)
        # return pos_ref_in_bins, pos_mov_in_bins

        pos_ref_selected_in_bins = []
        pos_mov_selected_in_bins = []

        # print(f"select_position_pairs at {datetime.now()}")
        for b in range(bins):
            pos_ref = pos_ref_in_bins[b]
            pos_mov = pos_mov_in_bins[b]

            pos_ref_selected, pos_mov_selected = M.select_position_pairs(
                img_ref_chnl, img_mov_chnl, pos_ref, pos_mov, w, T, q)

            pos_ref_selected_in_bins.append(pos_ref_selected)
            pos_mov_selected_in_bins.append(pos_mov_selected)

        # Merge the pairs together so that they are processed at once in subpixel matching
        pos_ref_selected_in_bins = np.vstack(pos_ref_selected_in_bins)
        pos_mov_selected_in_bins = np.vstack(pos_mov_selected_in_bins)

        # DEBUG: OK, but order is different. If compare with old version, pls uncomment this block.
        # idx_sorted = np.argsort(pos_ref_selected_in_bins[:, 0] * 1000 + pos_ref_selected_in_bins[:, 1])
        # pos_ref_selected_in_bins = pos_ref_selected_in_bins[idx_sorted]
        # pos_mov_selected_in_bins = pos_mov_selected_in_bins[idx_sorted]

        # print(f"pos_ref_selected_in_bins {pos_ref_selected_in_bins.shape}")
        # print(f"pos_mov_selected_in_bins {pos_mov_selected_in_bins.shape}")
        # return pos_ref_selected_in_bins, pos_mov_selected_in_bins

        # print(f"subpixel_match at {datetime.now()}")
        if prec_lvl > 0:
            blks_ref_in_bins, blks_mov_in_bins = M.subpixel_match(
                img_ref_chnl, img_mov_chnl, pos_ref_selected_in_bins, pos_mov_selected_in_bins,
                w, th, num_iter=prec_lvl)
        elif prec_lvl == 0:
            blks_ref = view_as_windows(img_ref_chnl, (w, w), step=(1, 1))
            blks_mov = view_as_windows(img_mov_chnl, (w, w), step=(1, 1))
            blks_ref_in_bins = blks_ref[pos_ref_selected_in_bins[:,
                                                                 0], pos_ref_selected_in_bins[:, 1]]
            blks_mov_in_bins = blks_mov[pos_mov_selected_in_bins[:,
                                                                 0], pos_mov_selected_in_bins[:, 1]]
            blks_ref_in_bins = blks_ref_in_bins.astype(np.float32)
            blks_mov_in_bins = blks_mov_in_bins.astype(np.float32)

        else:
            raise Exception("prec_lvl should be 0 or positive")

        # print("blks_ref_in_bins", blks_ref_in_bins.shape)
        # print("blks_mov_in_bins", blks_mov_in_bins.shape)

        # blks_ref_selected and blks_mov_selected are already sorted by their intensities
        blks_ref_in_bins = blks_ref_in_bins[:int(
            len(blks_ref_in_bins) // bins * bins)].reshape(bins, -1, w, w)
        blks_mov_in_bins = blks_mov_in_bins[:int(
            len(blks_mov_in_bins) // bins * bins)].reshape(bins, -1, w, w)

        # print(f"compute_variance_from_pairs at {datetime.now()}")
        for b in range(bins):
            blks_ref = blks_ref_in_bins[b]
            blks_mov = blks_mov_in_bins[b]

            intensity, variance = M.estimate_intensity_and_variance(
                blks_mov, blks_ref, T, 1, scale)

            # blks_ref_selected, blks_mov_selected = M.select_block_pairs(blks_ref, blks_mov, T, 1/3)
            # variance = M.compute_variance_from_pairs(blks_ref_selected, blks_mov_selected, T)
            # intensity = (np.mean(blks_ref_selected) + np.mean(blks_mov_selected)) / 2

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
