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


import cython
cimport cython

import numpy as np
cimport numpy as np

from skimage.util import view_as_windows, view_as_blocks
from scipy.fft import dctn, fft2, ifft2
# import cv2
# from scipy.signal import convolve2d
# from scipy import interpolate
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.fft import dctn

import os

########################################

ctypedef float T_t
T = np.float32

cdef extern from "limits.h":
    # cdef int INT32_MAX
    # cdef unsigned int UINT32_MAX
    cdef long INT64_MAX
    # cdef unsigned long UINT64_MAX

# cdef extern from "float.h":
    # cdef double DBL_MAX

########################################


def img_blur(img):
    """ Gaussian blur with typical 3x3 kernel

    Parameters
    ----------
    img: np.ndarray
        One channel input image
    
    Return
    ------
    img_blur: np.ndarray
        Blurred image in float
    """

    img_blur = gaussian_filter(img, sigma=1, mode="nearest")
    return img_blur


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve2d_sum(np.ndarray img, int h, int w):
    """ Given a raw image, compute the sums of overlapping blocks
        (See Algo. 2 of Sec. 5 in the paper)

    Parameters
    ----------
    img: np.ndarray
        Raw image
    h: int
        Height of block
    w: int
        Width of block        
    
    Return
    ------
    sum_of_blocks: np.ndarray
        Sums of pixel values of blocks, of size (H - h + 1, W - w + 1)

    """

    sum_of_blocks = np.cumsum(img, axis=1, dtype=np.float64)

    sum_of_blocks[:, w:] = sum_of_blocks[:, w:] - sum_of_blocks[:, :-w]
    sum_of_blocks = sum_of_blocks[:, w-1:]

    sum_of_blocks = np.cumsum(sum_of_blocks, axis=0, dtype=np.float64)
    sum_of_blocks[h:, :] = sum_of_blocks[h:, :] - sum_of_blocks[:-h, :]
    sum_of_blocks = sum_of_blocks[h-1:, :]

    return sum_of_blocks


@cython.boundscheck(False)
@cython.wraparound(False)
def pixel_match(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, int w, int th, int s):
    """ Dividing two images into wxw blocks. For each block in img_ref search the matched block in img_mov within a search range.
        (See Algo. 1 of Sec. 5 in the paper)

    Parameters
    ----------
    img_ref : numpy.ndarray
        Raw image at t, in size (H, W)
    img_mov : numpy.ndarray
        Raw image at t+1, in size (H, W)
    w : int
        Block size
    th : int
        Thickness of the ring bounding the patch
    s : int
        Search range for patch matching, the seach offset is within [-s, +s]

    Returns
    -------
    pos_ref: numpy.ndarray, dtype=int32
        2D positions of w*w blocks in img_ref, of size (N, 2)
    pos_mov: numpy.ndarray, dtype=int32
        2D positions of w*w blocks in img_mov, of size (N, 2)
    """

    assert img_ref.shape[0] == img_mov.shape[0]
    assert img_ref.shape[1] == img_mov.shape[1]

    cdef int H = img_ref.shape[0]
    cdef int W = img_ref.shape[1]

    cdef np.ndarray img_blur_ref = img_blur(img_ref)
    cdef np.ndarray img_blur_mov = img_blur(img_mov)

    cdef np.ndarray img_diff_offsets = np.zeros((2*s+1, 2*s+1, H-2*s, W-2*s), dtype=np.float32)
    
    cdef float[:, :, :, :] img_diff_offsets_view = img_diff_offsets
    cdef float[:, :] img_blur_ref_view = img_blur_ref
    cdef float[:, :] img_blur_mov_view = img_blur_mov

    cdef int off_i, off_j, i, j

    for off_i in range(0, 2*s+1):
        for off_j in range(0, 2*s+1):
            for i in range(0, H-2*s):
                for j in range(0, W-2*s):
                    img_diff_offsets_view[off_i, off_j, i, j] = img_blur_ref_view[i+s, j+s] - img_blur_mov_view[i+off_i, j+off_j]
                    img_diff_offsets_view[off_i, off_j, i, j] = img_diff_offsets_view[off_i, off_j, i, j] * img_diff_offsets_view[off_i, off_j, i, j]

    cdef int outer_sz = 2 * th + w
    cdef np.ndarray pos_ref = np.zeros(( (H-2*s-outer_sz+1) * (W-2*s-outer_sz+1), 2), dtype=np.int32)
    cdef np.ndarray pos_mov = np.zeros(( (H-2*s-outer_sz+1) * (W-2*s-outer_sz+1), 2), dtype=np.int32)
    
    cdef np.ndarray cost_of_offsets = np.zeros((2*s+1, 2*s+1, H-2*s-outer_sz+1, W-2*s-outer_sz+1), dtype=np.float64)
    
    for off_i in range(2*s+1):
        for off_j in range(2*s+1):
            cost_of_outer_blks = convolve2d_sum(img_diff_offsets[off_i, off_j], outer_sz, outer_sz) # (H - 2s - 2th - w + 1, ...)
            cost_of_inner_blks = convolve2d_sum(img_diff_offsets[off_i, off_j], w, w)[th:-th, th:-th] # (H - 2s - 2th - w + 1, ...)
            cost_of_offsets[off_i, off_j] = cost_of_outer_blks - cost_of_inner_blks

    assert np.all(cost_of_offsets >= 0)

    cdef np.int32_t[:, :] pos_ref_view = pos_ref
    cdef np.int32_t[:, :] pos_mov_view = pos_mov
    cdef np.float64_t[:, :, :, :] cost_of_offsets_view = cost_of_offsets

    cdef np.float64_t cost_best
    cdef int off_i_best
    cdef int off_j_best
    cdef int nb_pos = 0

    cdef int H_img_blk = H-2*s-outer_sz+1
    cdef int W_img_blk = W-2*s-outer_sz+1

    for i in range(H_img_blk):
        for j in range(W_img_blk):
            cost_best = INT64_MAX
            off_i_best = 0
            off_j_best = 0
            for off_i in range(2*s+1):
                for off_j in range(2*s+1):
                    if cost_best > cost_of_offsets_view[off_i, off_j, i, j]:
                        cost_best = cost_of_offsets_view[off_i, off_j, i, j]
                        off_i_best = off_i; off_j_best = off_j
            
            nb_pos = i * W_img_blk + j
            pos_ref_view[nb_pos, 0] = i + s + th
            pos_ref_view[nb_pos, 1] = j + s + th
            pos_mov_view[nb_pos, 0] = pos_ref_view[nb_pos, 0] + (off_i_best - s)
            pos_mov_view[nb_pos, 1] = pos_ref_view[nb_pos, 1] + (off_j_best - s)

    return pos_ref, pos_mov


cdef inline float compute_low_freq_energy(np.float32_t[:, :] D, int w, int T):
    """ Compute the low frequency energy of a DCT block
        (See Algo. 6 of Sec. 5 in the paper)
    
    Parameters
    ----------
    D: np.ndarray
        DCT block, of size (w, w)
    w: int
        Block size
    T: int
        threshold for separating the entries for low and high frequency DCT coefficents
    
    Return
    ------
    e: float
        low frequency energy
    """

    cdef float e = 0
    cdef int i, j

    for i in range(w):
        for j in range(w):
            if i + j <= T:
                e = e + D[i, j] * D[i, j]
    return e


@cython.boundscheck(False)
@cython.wraparound(False)
def select_position_pairs(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, \
        np.ndarray[np.int32_t, ndim=2] pos_ref, np.ndarray[np.int32_t, ndim=2] pos_mov, int w, int T, float q):
    """ Select a percentile of block pairs whose difference have the least low-frequency energies
        (See Algo. 4 of Sec. 5 in the paper)

    Parameters
    ----------
    img_ref: np.ndarray
        Reference image of size (H, W)
    img_mov: np.ndarray
        Moving image of size (H, W)
    pos_ref: np.ndarray
        Block positions in reference image, of size (N, 2)
    pos_mov: np.ndarray
        Block positions in moving image, of size (N, 2)
    w: int
        Block size
    T: int
        Threshold for separating the entries for low and high frequency DCT coefficents
    q: float
        percentile of selected blocks, in [0, 1]

    Returns
    -------
    pos_selected_ref: np.ndarray
        Selected block positions in reference image, of size (M, 2),
        with M = floor(N * q)
    pos_selected_mov: np.ndarray
        Selected block positions in moving image, of size (M, 2)
    """

    blks_ref = view_as_windows(img_ref, (w, w), step=(1, 1))[pos_ref[:,0], pos_ref[:, 1]] # (N, w, w)
    blks_mov = view_as_windows(img_mov, (w, w), step=(1, 1))[pos_mov[:,0], pos_mov[:, 1]] # (N, w, w)

    cdef int N = pos_ref.shape[0]
    cdef np.ndarray E = np.zeros(N, dtype=np.float32)
    cdef np.float32_t[:] E_view = E
    
    dct_blks_diff = dctn(blks_ref - blks_mov, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)
    dct_blks_diff = dct_blks_diff.astype(np.float32)

    cdef np.float32_t[:, :, :] dct_blks_diff_view = dct_blks_diff
    cdef np.float32_t[:, :] dct_blk_view

    cdef int i
    for i in range(N):
        dct_blk_view = dct_blks_diff_view[i]
        E_view[i] = compute_low_freq_energy(dct_blk_view, w, T)

    I = np.argsort(E)[:int(N * q)]

    pos_selected_ref = pos_ref[I]
    pos_selected_mov = pos_mov[I]

    return pos_selected_ref, pos_selected_mov


@cython.boundscheck(False)
@cython.wraparound(False)
def select_block_pairs(np.ndarray[np.float32_t, ndim=3] blks_ref, np.ndarray[np.float32_t, ndim=3] blks_mov, int T, float q):
    """ Select a percentile of block pairs whose difference have the least low-frequency energies

    Parameters
    ----------
    blks_ref: np.ndarray
        Blocks in reference image of size (N, w, w)
    blks_mov: np.ndarray
        Blocks in moving image of size (N, w, w)
    T: int
        Threshold for separating the entries for low and high frequency DCT coefficents
    q: float
        percentile of selected blocks

    Returns
    -------
    pos_selected_ref: np.ndarray
        Selected blocks in reference image, of size (M, w, w),
        with M = floor(N * q)
    pos_selected_mov: np.ndarray
        Selected blocks in moving image, of size (M, w, w)
    """

    cdef int N = blks_mov.shape[0]
    cdef int w = blks_mov.shape[1]

    cdef np.ndarray E = np.zeros(N, dtype=np.float32)
    cdef np.float32_t[:] E_view = E
    
    dct_blks_diff = dctn(blks_ref - blks_mov, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)

    cdef np.float32_t[:, :, :] dct_blks_diff_view = dct_blks_diff
    cdef np.float32_t[:, :] dct_blk_view

    cdef int i
    for i in range(N):
        dct_blk_view = dct_blks_diff_view[i]
        E_view[i] = compute_low_freq_energy(dct_blk_view, w, T)

    I = np.argsort(E)[:int(N * q)]

    blks_selected_ref = blks_ref[I]
    blks_selected_mov = blks_mov[I]

    return blks_selected_ref, blks_selected_mov


@cython.boundscheck(False)
@cython.wraparound(False)
def partition_obsolete(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, \
              np.ndarray[np.int32_t, ndim=2] pos_ref, np.ndarray[np.int32_t, ndim=2] pos_mov, int w, int b):
    """ Partition the block pairs into bins according to their mean intensities
        (See Algo. 3 of Sec. 5 in the paper)

    Parameters
    ----------
    img_ref: np.ndarray
        Reference image of size (H, W)
    img_mov: np.ndarray
        Moving image of size (H, W)
    pos_ref: np.ndarray
        Block positions in reference image, of size (N, 2)
    pos_mov: np.ndarray
        Block positions in moving image, of size (N, 2)
    w: int
        Block size
    b : int
        number of bins

    Returns
    -------
    bins_pos_ref: np.ndarray
        Bins of block positions in reference image, of size (b, M, 2), with M
        the number of blocks per bin
    bins_pos_mov: np.ndarray
        Bins of block positions in moving image, of size (b, M, 2)
    """
    
    blks_ref = view_as_windows(img_ref, (w, w), step=(1, 1))[pos_ref[:,0], pos_ref[:, 1]] # (N, w, w)
    blks_mov = view_as_windows(img_mov, (w, w), step=(1, 1))[pos_mov[:,0], pos_mov[:, 1]] # (N, w, w)

    L = (np.mean(blks_ref, axis=(1, 2)) + np.mean(blks_mov, axis=(1, 2))) / 2 # (N, )
    
    I = np.argsort(L)
    I_in_bins = I[:len(L) // b * b].reshape(b, -1)

    bins_pos_ref = pos_ref[I_in_bins]
    bins_pos_mov = pos_mov[I_in_bins]

    return bins_pos_ref, bins_pos_mov


@cython.boundscheck(False)
@cython.wraparound(False)
def partition(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, \
              np.ndarray[np.int32_t, ndim=2] pos_ref, np.ndarray[np.int32_t, ndim=2] pos_mov, int w, int b):
    """ Partition the block pairs into bins according to their mean intensities
        (See Algo. 3 of Sec. 5 in the paper)

    Parameters
    ----------
    img_ref: np.ndarray
        Reference image of size (H, W)
    img_mov: np.ndarray
        Moving image of size (H, W)
    pos_ref: np.ndarray
        Block positions in reference image, of size (N, 2)
    pos_mov: np.ndarray
        Block positions in moving image, of size (N, 2)
    w: int
        Block size
    b : int
        number of bins

    Returns
    -------
    bins_pos_ref: np.ndarray
        Bins of block positions in reference image, of size (b, M, 2), with M
        the number of blocks per bin
    bins_pos_mov: np.ndarray
        Bins of block positions in moving image, of size (b, M, 2)
    """
    sum_conv_ref = convolve2d_sum(img_ref, w, w) 
    sum_conv_mov = convolve2d_sum(img_mov, w, w)

    mean_blks_ref = sum_conv_ref[pos_ref[:,0], pos_ref[:, 1]] / w**2
    mean_blks_mov = sum_conv_mov[pos_mov[:,0], pos_mov[:, 1]] / w**2

    L = (mean_blks_ref + mean_blks_mov) / 2 # (N, )

    I = np.argsort(L)
    I_in_bins = I[:len(L) // b * b].reshape(b, -1)

    bins_pos_ref = pos_ref[I_in_bins]
    bins_pos_mov = pos_mov[I_in_bins]

    return bins_pos_ref, bins_pos_mov


# Interaction between numpy and native c
# https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html

np.import_array()

cdef extern from "helper.h":
    void find_best_matching(
            float *img_ref, float *img_mov, np.uint16_t *pos_ref, np.uint16_t *pos_mov_init, \
            int H, int W, int N, int w, int th,int ups_factor, np.uint16_t *pos_mov_final)

    void find_best_matching_one_shot(
            float *img_ref, float *img_mov_ups, np.uint16_t *pos_ref, np.uint16_t *pos_mov_init, \
            int H, int W, int N, int w, int th,int ups_factor, np.uint16_t *pos_mov_final)


@cython.boundscheck(False)
@cython.wraparound(False)
def find_best_matching_func(
        np.ndarray[float, ndim=2, mode="c"] img_ref not None,
        np.ndarray[float, ndim=2, mode="c"] img_mov not None,
        np.ndarray[np.uint16_t, ndim=2, mode="c"] pos_ref not None,
        np.ndarray[np.uint16_t, ndim=2, mode="c"] pos_mov_init not None,
        int w, int th, int ups_factor
    ):
    """ Wrap C interface into python interface """

    cdef int N = pos_ref.shape[0]
    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] pos_mov = np.zeros((N, 2), dtype=np.uint16)

    # find_best_matching / find_best_matching_one_shot
    find_best_matching(<float*> np.PyArray_DATA(img_ref),
                       <float*> np.PyArray_DATA(img_mov),
                       <np.uint16_t*> np.PyArray_DATA(pos_ref),
                       <np.uint16_t*> np.PyArray_DATA(pos_mov_init),
                       img_ref.shape[0], img_ref.shape[1],
                       N, w, th, ups_factor,
                       <np.uint16_t*> np.PyArray_DATA(pos_mov))
    
    return pos_mov


@cython.boundscheck(False)
@cython.wraparound(False)
def upsample_image(img_in, ups_factor):
    """ Upsample an image by padding zero at the two sides of the centered
        fourier transforms of the input image.

    Parameters
    ----------
    img_in : ndarray
        Original image, of size (H, W)
    ups_factor : int
        Upsample factor

    Returns
    -------
    img_ups : ndarray
        Shifted image
    """

    H, W = img_in.shape
    H_obj = ups_factor * H
    W_obj = ups_factor * W

    img_in = img_in / 255.0
    fft = np.fft.fftshift(fft2(img_in, norm="forward", workers=os.cpu_count()))
    fr = fft.real
    fi = fft.imag

    fr = np.pad(fr, ( ((H_obj - H) // 2, ), ((W_obj - W) // 2,) ), 'constant', constant_values=0)
    fi = np.pad(fi, ( ((H_obj - H) // 2, ), ((W_obj - W) // 2,) ), 'constant', constant_values=0)
    
    fft_ups = np.fft.ifftshift(fr + fi * 1j)

    img_ups = ifft2(fft_ups, norm="forward", workers=os.cpu_count()).real

    img_ups = img_ups * 255.0
    img_ups = img_ups.astype(np.float32)

    return img_ups


@cython.boundscheck(False)
@cython.wraparound(False)
def blocks_from_image(img, blk_sz, pos):
    """
    Parameters
    ----------
    img: ndarray
        Image of size (H, W)
    blk_sz: int
        Block size, suppose the blocks are squared.
    pos: ndarray
        Positions of the blocks, of size (N, 2)

    Return
    ------
    blocks: ndarray
        Selected blocks according to the block positions, of size (N, blk_sz, blk_sz)
    """

    blocks = view_as_windows(img, (blk_sz, blk_sz), step=(1, 1))[pos[:, 0], pos[:, 1]]
    return blocks
    

@cython.boundscheck(False)
@cython.wraparound(False)
def subpixel_match(img_ref, img_mov, pos_ref, pos_mov_init, w: int, th: int, order:int=2):
    """ Compute the blocks in the moving image that match the blocks in the reference image
        by matching their surrounding rings in subpixelic precision

    Parameters
    ----------
    img_ref: ndarray
        Reference image, of size (H, W)
    img_mov: ndarray
        Moving image, of size (H, W)
    pos_ref: ndarray
        Positions of w*w blocks in the reference image, of size (N, 2).
    pos_mov_init : ndarray
        Initial shifts of w*w blocks in the moving image, of size (N, 2).
    w: int
        Block size
    th: int
        Thickness of surrounding ring
        at each iteration (e.g. 1/2px -> 1/4px -> 1/8px etc.)
    order: int
        The scale of upsampling the moving image.

    Returns
    -------
    blocks_ref: np.ndarray, dtype=float32
        Matched patches in the reference image, of size (N, w, w)
    blocks_mov: np.ndarray, dtype=float32
        Matched patches in the moving image, of size (N, w, w)
    """ 

    assert img_ref.shape == img_mov.shape
    H, W = img_ref.shape

    assert pos_ref.shape == pos_mov_init.shape
    N, _ = pos_ref.shape

    w_ups = w

    # TODO: add to IPOL paper
    # remove blocks within 4 pixels from the border
    cdef int margin = 4
    valid_mask = (pos_mov_init[:, 0] >= th + margin) & (pos_mov_init[:, 0] < H - w + 1 - th - margin) & \
                     (pos_mov_init[:, 1] >= th + margin) & (pos_mov_init[:, 1] < W - w + 1 - th - margin)
    pos_ref = pos_ref[valid_mask]
    
    pos_mov_ups = pos_mov_init[valid_mask]

    # Start matching
    ups_factor = 2 ** order

    img_blur_ref = img_blur(img_ref)
    img_blur_mov = img_blur(img_mov)

    img_blur_mov_ups = upsample_image(img_blur_mov, ups_factor)
    pos_mov_ups = pos_mov_ups * ups_factor

    w_ups = w * ups_factor

    
    if (not img_blur_ref.flags["C_CONTIGUOUS"]) or (img_blur_ref.dtype != np.float32):
        img_blur_ref = np.ascontiguousarray(img_blur_ref, dtype=np.float32)
    if (not img_blur_mov_ups.flags["C_CONTIGUOUS"]) or (img_blur_mov_ups.dtype != np.float32):
        img_blur_mov_ups = np.ascontiguousarray(img_blur_mov_ups, dtype=np.float32)
    if (not pos_ref.flags["C_CONTIGUOUS"]) or (pos_ref.dtype != np.uint16):
        pos_ref = np.ascontiguousarray(pos_ref, dtype=np.uint16)
    if (not pos_mov_ups.flags["C_CONTIGUOUS"]) or (pos_mov_ups.dtype != np.uint16):
        pos_mov_ups = np.ascontiguousarray(pos_mov_ups, dtype=np.uint16)

    pos_mov_ups = find_best_matching_func(img_blur_ref, img_blur_mov_ups, pos_ref, pos_mov_ups, w, th, ups_factor)
    
    img_mov_ups = upsample_image(img_mov, ups_factor)
    
    inner_blocks_ref = blocks_from_image(img_ref, w, pos_ref)
    inner_blocks_mov = blocks_from_image(img_mov_ups, w_ups, pos_mov_ups)
    inner_blocks_mov = inner_blocks_mov[:, ::ups_factor, ::ups_factor]

    return inner_blocks_ref.astype(np.float32), inner_blocks_mov.astype(np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_variance_from_pairs(blks_ref, blks_mov, T, factor):
    """ Compute noise variance from block pairs
        (See Algo. 5 of Sec. 5 in the paper)

    Parameters
    ----------
    blks_ref: ndarray
        Blocks in reference image of size (N, w, w)
    blks_mov: ndarray
        Blocks in moving image of size (N, w, w)
    T: int
        Threshold for separating the entries for low and high frequency DCT coefficents
    factor: int
        Downscaling factor.

    Returns
    -------
    variance: float
        Noise variance
    """

    assert blks_ref.shape == blks_mov.shape
    cdef int N = blks_ref.shape[0]
    cdef int w = blks_ref.shape[1]

    blks_diff = blks_mov - blks_ref

    # Subscaling the block differences, see Algo. 5 of Sec. 5 in the paper
    if factor > 1:
        N, W, _ = blks_ref.shape

        assert W % factor == 0, "The block size must be multiple of the subsample factor"
        w = W // factor

        # blks_diff = view_as_blocks(blks_diff, (1, factor, factor)).squeeze() # (N, w, w, factor, factor)
        # blks_diff.transpose((0, 3, 4, 1, 2))
        # blks_diff = blks_diff.reshape(-1, w, w)

        blks_diff = view_as_blocks(blks_diff, (1, factor, factor)).squeeze().mean(axis=(-1, -2)) # (N, w, w, factor, factor)
    
    dct_blks_diff = dctn(blks_diff, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)

    VH = []
    cdef int i, j
    for i in range(w):
        for j in range(w):
            if i + j > T:
                VH.append(np.mean(dct_blks_diff[:, i, j] ** 2))
    
    return np.median( np.array(VH) ) / 2


@cython.boundscheck(False)
@cython.wraparound(False)
def remove_saturated_obsolete(np.ndarray img_ref, np.ndarray img_mov,
                     np.ndarray pos_ref, np.ndarray pos_mov, int w):
    """ Remove saturated block pairs and return non-saturated pairs
    
    Parameters
    ----------
    img_ref: np.ndarray
        Reference image
    img_mov: np.ndarray
        Moving image
    pos_ref: np.ndarray
        Block positions in reference image
    pos_mov: np.ndarray
        Block positions in moving image
    w: int
        Block size

    Return
    ------
    [0]: np.ndarray
        Positions of non-saturated blocks in reference frame
    [1]: np.ndarray
        Positions of non-saturated blocks in moving frame
    """

    blks_ref = view_as_windows(img_ref, (w,w), step=(1,1))[pos_ref[:, 0], pos_ref[:, 1]]
    blks_mov = view_as_windows(img_mov, (w,w), step=(1,1))[pos_mov[:, 0], pos_mov[:, 1]]

    max_val = np.max([np.max(img_ref), np.max(img_mov)])

    valid_mask_ref = np.max(blks_ref, axis=(1, 2)) != max_val
    valid_mask_mov = np.max(blks_mov, axis=(1, 2)) != max_val
    valid_mask = valid_mask_ref & valid_mask_mov

    return pos_ref[valid_mask], pos_mov[valid_mask]



@cython.boundscheck(False)
@cython.wraparound(False)
def remove_saturated(np.ndarray img_ref, np.ndarray img_mov,
                     np.ndarray pos_ref, np.ndarray pos_mov, int w):
    """ Remove saturated block pairs and return non-saturated pairs
    
    Parameters
    ----------
    img_ref: np.ndarray
        Reference image
    img_mov: np.ndarray
        Moving image
    pos_ref: np.ndarray
        Block positions in reference image, of size (N, 2)
    pos_mov: np.ndarray
        Block positions in moving image, of size (N, 2)
    w: int
        Block size

    Return
    ------
    [0]: np.ndarray
        Positions of non-saturated blocks in reference frame
    [1]: np.ndarray
        Positions of non-saturated blocks in moving frame
    """

    H, W = img_ref.shape[0], img_ref.shape[1]
    max_val = np.max([np.max(img_ref), np.max(img_mov)])

    max_conv_ref = maximum_filter(img_ref, w) # (H, W)
    max_conv_ref = max_conv_ref[w//2: w//2+H-w+1, w//2: w//2+W-w+1]

    max_conv_mov = maximum_filter(img_ref, w) # (H, W)
    max_conv_mov = max_conv_mov[w//2: w//2+H-w+1, w//2: w//2+W-w+1]

    # valid_img_ref = max_conv_ref != max_val
    # valid_img_mov = max_conv_mov != max_val

    valid_mask_ref = max_conv_ref[pos_ref[:,0], pos_ref[:,1]] != max_val
    valid_mask_mov = max_conv_mov[pos_mov[:,0], pos_mov[:,1]] != max_val
    valid_mask = valid_mask_ref & valid_mask_mov

    return pos_ref[valid_mask], pos_mov[valid_mask]


    # valid_mask_ref = (img_ref < max_val)
    # cdef bool[:] valid_mask_ref_view = valid_mask_ref
    # cdef int i, j
    # for i in range(H):
    #     for j in range(W):
    #         if 


    # blks_ref = view_as_windows(img_ref, (w,w), step=(1,1))[pos_ref[:, 0], pos_ref[:, 1]]
    # blks_mov = view_as_windows(img_mov, (w,w), step=(1,1))[pos_mov[:, 0], pos_mov[:, 1]]

    # max_val = np.max([np.max(img_ref), np.max(img_mov)])

    # valid_mask_ref = np.max(blks_ref, axis=(1, 2)) != max_val
    # valid_mask_mov = np.max(blks_mov, axis=(1, 2)) != max_val
    # valid_mask = valid_mask_ref & valid_mask_mov

    # return pos_ref[valid_mask], pos_mov[valid_mask]


@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_intensity_and_variance(blks_ref, blks_mov,
                                    T: int, q: float):
    """ Select block pairs of a bin and compute an intensity and a noise variance

    Parameters
    ----------
    blks_ref: np.ndarray
        blocks in reference frame, in shape (N, w, w)
    blks_mov: np.ndarray
        blocks in moving frame, in shape (N, w, w)
    T: int
        frequency separator
    q: float
        quantile of selected block pairs
    fact: int
        Subscaling factor. 1 for non subscaling

    Return
    ------
    intensity: float
        mean intensity of the selected block pairs
    variance: float
        noise variance from the selected inter-block differences
    """

    N, w, _ = blks_ref.shape

    blks_diff = np.array(blks_mov - blks_ref)
    dct_blks = dctn(blks_diff, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)
    
    # Low-frequency energy
    cdef np.ndarray E = np.zeros(N, dtype=np.float32)
    cdef np.float32_t[:] E_view = E
    cdef np.float32_t[:, :, :] dct_blks_view = dct_blks

    cdef int i
    for i in range(N):
        E_view[i] = compute_low_freq_energy(dct_blks_view[i], w, T)

    I = np.argsort(E)[:int(N * q)]
    dct_blks = dct_blks[I] # (N*q, w, w)
    VH = []
    for i in range(w):
        for j in range(w):
            if i + j > T:
                VH.append(np.mean(dct_blks[:, i, j] ** 2))
    
    variance = np.median( np.array(VH) ) / 2
    intensity = (blks_mov.mean() + blks_ref.mean()) / 2

    return intensity, variance