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


import cython
cimport cython

import numpy as np
cimport numpy as np

import cv2
from skimage.util import view_as_windows, view_as_blocks

from scipy.fft import dct, dctn, ifft2

# For debug use
from inspect import currentframe

########################################

ctypedef np.int32_t T_t
T = np.int32

cdef extern from "limits.h":
    cdef int INT32_MAX
    cdef unsigned int UINT32_MAX
    cdef long INT64_MAX
    cdef unsigned long UINT64_MAX 

########################################

@cython.boundscheck(False)
@cython.wraparound(False)
def img_direction(img, k):
    """ Divide 360 degrees into k ranges, compute the gradient directions of each pixel and 
        map with the directions with an integer in [0, k-1].
        (See algo. 1 of sec. 5.1 in the paper)

    Parameters
    ----------
    img: np.ndarray
        One channel input image
    k: int
        Number of areas to divide in 360 degrees
    
    Return
    ----------
    img_dir: np.ndarray
        Image of uint8 values in [0, k-1]. Each int value indicates a gradient direction.

    """

    img_dx = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 1, 0, 3)
    img_dy = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 0, 1, 3)
    img_angle = np.arctan2(img_dy, img_dx) # [-pi, pi)
    img_dir = np.floor(img_angle / 2 / np.pi * k)

    img_dir[img_dir < 0] += k
    img_dir = img_dir.astype(np.int8)

    return img_dir


@cython.boundscheck(False)
@cython.wraparound(False)
def integral_image(T_t[:, :] img, int H, int W):
    """ Compute integral image
        (See algo. 3 of sec. 5.1 in the paper)

    Parametes
    ---------
    img: np.ndarray
        input image, of size (H, W)
    H: int
        Height
    W: int
        Width

    Return
    ------
    img_int: np.ndarray
        output integral image, of size (H, W)

    """

    cdef np.ndarray img_int = np.zeros((H, W), dtype=T)
    img_int[0] = np.cumsum(img[0])

    cdef int s
    cdef int i, j

    cdef T_t[:, :] img_view = img
    cdef T_t[:, :] img_int_view = img_int


    for i in range(1, H):
        s = img_view[i, 0]
        img_int_view[i, 0] = img_int_view[i-1, 0] + s
        for j in range(1, W):
            s = s + img_view[i, j]
            img_int_view[i, j] = img_int_view[i-1, j] + s

    return img_int


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.uint32_t compute_cost(np.uint32_t[:, :] img_int, int pi, int pj, int w):
    """ Compute the sum of costs in a block using integral image
        (See algo. 4 of sec. 5.1 in the paper)
    Parameters
    ----------
    img_int: np.ndarray
        Integral image
    pi: int
        Row of the top-left pixel of the block, of size (2, )
    pj: int
        Column of the top-left pixel of the block, of size (2, )
    w: int
        Block size
    
    Return
    ------
    cost: int
        Sum of pixel values in the block

    """
    
    cdef np.uint32_t cost
    if pi > 0 and pj > 0:
        cost = img_int[pi + w - 1, pj + w - 1] + img_int[pi - 1, pj - 1] - img_int[pi - 1, pj + w -1] - img_int[pi + w - 1, pj - 1]
    elif pi > 0 and pj == 0:
        cost = img_int[pi + w - 1, pj + w - 1] - img_int[pi - 1, pj + w - 1]
    elif pi == 0 and pj > 0:
        cost = img_int[pi + w - 1, pj + w - 1] - img_int[pi + w - 1, pj - 1]
    else:
        cost = img_int[pi + w - 1, pj + w - 1]


    return cost


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve2d_sum(np.ndarray img, int h, int w):
    """ Given a raw image, compute the sums of overlapping blocks

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

    sum_of_blocks = np.cumsum(img, axis=1, dtype=np.uint16)

    sum_of_blocks[:, w:] = sum_of_blocks[:, w:] - sum_of_blocks[:, :-w]
    sum_of_blocks = sum_of_blocks[:, w-1 : ]

    sum_of_blocks = np.cumsum(sum_of_blocks, axis=0)
    sum_of_blocks[h:, :] = sum_of_blocks[h:, :] - sum_of_blocks[:-h, :]
    sum_of_blocks = sum_of_blocks[h-1:, :]

    return sum_of_blocks


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int dist(np.int8_t x, np.int8_t y, int k):
    cdef int d0, d1
    if x > y: d0 = x - y
    else: d0 = y - x
    d1 = k - d0
    return d0 if d0 < d1 else d1


@cython.boundscheck(False)
@cython.wraparound(False)
def pixel_match(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, int w, int th, int s, int k):
    """ Dividing two images into wxw blocks. For each block in img_ref search the matched block in img_mov within a search range.
        (See algo. 2 of sec. 5.2 in the paper)

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
    k : int
        Number of divided areas of 360 degrees

    Returns
    -------
    pos_ref: numpy.ndarray, dtype=int32
        2D positions of w*w blocks in img_ref, of size (N, 2)
    pos_mov: numpy.ndarray, dtype=int32
        2D positions of w*w blocks in img_mov, of size (N, 2)
    """

    # print(img_ref.shape)
    # print(img_mov.shape)
    assert img_ref.shape[0] == img_mov.shape[0]
    assert img_ref.shape[1] == img_mov.shape[1]

    cdef int H = img_ref.shape[0]
    cdef int W = img_ref.shape[1]

    cdef np.ndarray img_dir_ref = img_direction(img_ref, k)
    cdef np.ndarray img_dir_mov = img_direction(img_mov, k)

    # cdef np.ndarray offsets = np.zeros((2*s+1, 2*s+1), dtype=np.int64)
    cdef np.ndarray img_diff_offsets = np.zeros((2*s+1, 2*s+1, H-2*s, W-2*s), dtype=np.uint32)
    
    cdef np.uint32_t[:, :, :, :] img_diff_offsets_view = img_diff_offsets
    cdef np.int8_t[:, :] img_dir_ref_view = img_dir_ref
    cdef np.int8_t[:, :] img_dir_mov_view = img_dir_mov
    
    cdef np.uint32_t[:, :] img_diff_view

    
    cdef int off_i, off_j, i, j
    for off_i in xrange(0, 2*s+1):
        for off_j in xrange(0, 2*s+1):
            img_diff_view = img_diff_offsets_view[off_i, off_j]
            for i in xrange(0, H-2*s):
                for j in xrange(0, W-2*s):
                    # TODO: update k in paper
                    img_diff_view[i, j] = dist(img_dir_ref_view[i+s, j+s], img_dir_mov_view[i+off_i, j+off_j], k)

    cdef int outer_sz = 2 * th + w
    cdef np.ndarray pos_ref = np.zeros(( (H-2*s-outer_sz+1) * (W-2*s-outer_sz+1), 2), dtype=np.int32)
    cdef np.ndarray pos_mov = np.zeros(( (H-2*s-outer_sz+1) * (W-2*s-outer_sz+1), 2), dtype=np.int32)
    
    cdef np.ndarray cost_of_offsets = np.zeros((2*s+1, 2*s+1, H-2*s-outer_sz+1, W-2*s-outer_sz+1), dtype=np.int32)
    # cdef np.ndarray sum_of_outer_blks, sum_of_inner_blks
    for off_i in xrange(2*s+1):
        for off_j in xrange(2*s+1):
            cost_of_outer_blks = convolve2d_sum(img_diff_offsets[off_i, off_j], outer_sz, outer_sz) # (H - 2s - 2th - w + 1, ...)
            cost_of_inner_blks = convolve2d_sum(img_diff_offsets[off_i, off_j], w, w)[th:-th, th:-th] # (H - 2s - 2th - w + 1, ...)
            cost_of_offsets[off_i, off_j] = cost_of_outer_blks - cost_of_inner_blks

    assert np.all(cost_of_offsets >= 0)

    cdef np.int32_t[:, :] pos_ref_view = pos_ref
    cdef np.int32_t[:, :] pos_mov_view = pos_mov
    cdef np.int32_t[:, :, :, :] cost_of_offsets_view = cost_of_offsets

    cdef np.int32_t cost_best
    cdef int off_i_best, off_j_best
    cdef int nb_pos = 0

    for i in range(H-2*s-outer_sz+1):
        for j in range(W-2*s-outer_sz+1):
            cost_best = INT32_MAX
            off_i_best = 0
            off_j_best = 0
            for off_i in range(2*s+1):
                for off_j in range(2*s+1):
                    if cost_best > cost_of_offsets_view[off_i, off_j, i, j]:
                        cost_best = cost_of_offsets_view[off_i, off_j, i, j]
                        off_i_best = off_i; off_j_best = off_j
            pos_ref_view[nb_pos, 0] = i + s + th
            pos_ref_view[nb_pos, 1] = j + s + th
            pos_mov_view[nb_pos, 0] = pos_ref_view[nb_pos, 0] + (off_i_best - s)
            pos_mov_view[nb_pos, 1] = pos_ref_view[nb_pos, 1] + (off_j_best - s)

            nb_pos = nb_pos + 1

    return pos_ref, pos_mov


@cython.boundscheck(False)
@cython.wraparound(False)
def pixel_match_obsolete(img_ref, img_mov, w, th, s, k):
    """ Dividing two images into wxw blocks. For each block in img_ref search the matched block in img_mov within a search range.
        (See algo. 2 of sec. 5.2 in the paper)

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
    k : int
        Number of divided areas of 360 degrees

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        [0]: Raw blocks in img_ref, of size (N, w, w), N = (H - w + 1 - 2 * search_range) * (W - w + 1 - 2 * search_range)
        [1]: Raw blocks in img_mov that match the blocks in img_ref, of size (N, w, w)
    """

    
    H, W = img_ref.shape

    img_ref_dir = img_direction(img_ref, k)
    img_mov_dir = img_direction(img_mov, k)

    img_ref_dir_cropped = img_ref_dir[s:(H-s), s:(W-s)]

    img_dir_diff_cdds = np.zeros(((2*s+1)**2, (H-s*2), (W-s*2)), dtype=np.uint8)
    idx = 0
    for i in range(-s, s+1):
        for j in range(-s, s+1):
            img_mov_dir_cropped = img_mov_dir[(s+i):(H-s+i), (s+j):(W-s+j)]
            img_dir_diff_cropped = np.abs(img_ref_dir_cropped - img_mov_dir_cropped)
            img_dir_diff_cropped = np.minimum(img_dir_diff_cropped, k - img_dir_diff_cropped)
            
            img_dir_diff_cdds[idx] = img_dir_diff_cropped
            idx += 1

    sz_patch = 2*th + w
    scores_of_blocks_all_offsets = np.zeros(((2*s+1)**2, (H-s*2-sz_patch+1), (W-s*2-sz_patch+1)))
    
    
    for idx in range((2*s+1) ** 2):
        scores_of_blocks = np.cumsum(img_dir_diff_cdds[idx], axis=1, dtype=np.uint16)
        scores_of_blocks_exclude = scores_of_blocks.copy()

        scores_of_blocks[:, sz_patch:] = scores_of_blocks[:, sz_patch:] - scores_of_blocks[:, :-sz_patch]
        scores_of_blocks = scores_of_blocks[:, sz_patch-1 : ]

        scores_of_blocks = np.cumsum(scores_of_blocks, axis=0)
        scores_of_blocks[sz_patch:, :] = scores_of_blocks[sz_patch:, :] - scores_of_blocks[:-sz_patch, :]
        scores_of_blocks = scores_of_blocks[sz_patch-1 :, :]

        
        scores_of_blocks_exclude[:, w:] = scores_of_blocks_exclude[:, w:] - scores_of_blocks_exclude[:, :-w]
        scores_of_blocks_exclude = scores_of_blocks_exclude[:, w-1 : ]

        scores_of_blocks_exclude = np.cumsum(scores_of_blocks_exclude, axis=0)
        scores_of_blocks_exclude[w:, :] = scores_of_blocks_exclude[w:, :] - scores_of_blocks_exclude[:-w, :]
        scores_of_blocks_exclude = scores_of_blocks_exclude[w-1 :, :]

        scores_of_blocks_all_offsets[idx] = scores_of_blocks - scores_of_blocks_exclude[th:-th, th:-th]


    match_indices = np.argmin(scores_of_blocks_all_offsets, axis=0) # N
    # match_indices = match_indices.reshape(1, (H-s*2-sz_patch+1), (W-s*2-sz_patch+1), 1, 1)
    match_indices = match_indices.reshape(-1)

    # costs_best = np.min(scores_of_blocks_all_offsets, axis=0).reshape(-1)

    
    # arr_2d = np.arange((H-s*2-w+1) * (W-s*2-w+1)).reshape(H-s*2-w+1, W-s*2-w+1)
    iv, jv = np.meshgrid(np.arange(H-s*2-sz_patch+1), np.arange(W-s*2-sz_patch+1), indexing='ij')

    arr_2d = (iv * (W-sz_patch+1) + jv)
    # print(arr_2d.shape)
    match_indices_final = arr_2d.reshape(-1) + match_indices // (2*s+1) * (W-sz_patch+1) + match_indices % (2*s+1)

    img_mov_blocks = view_as_windows(img_mov, (sz_patch, sz_patch), step=(1, 1)).reshape(-1, sz_patch, sz_patch)
    img_mov_blocks_best = img_mov_blocks[match_indices_final]

    
    
    img_ref_cropped_blocks = view_as_windows(img_ref[s:(H-s), s:(W-s)], (sz_patch, sz_patch), step=(1, 1)).reshape(-1, sz_patch, sz_patch)

    print("patch match done")
    # return img_ref_cropped_blocks[..., th:(w+th), th:(w+th)], img_mov_blocks_best[..., th:(w+th), th:(w+th)]


    match_indices = np.argmin(scores_of_blocks_all_offsets, axis=0)
    shift_i = match_indices // (2*s+1) - s
    shift_j = match_indices % (2*s+1) - s

    # print(match_indices.shape)
    iv, jv = np.meshgrid(np.arange(H-s*2-sz_patch+1), np.arange(W-s*2-sz_patch+1), indexing='ij')
    iv += s + shift_i; jv += s + shift_j
    pos_mov = np.array([iv.reshape(-1), jv.reshape(-1)]).T + th

    iv, jv = np.meshgrid(np.arange(H-w+1), np.arange(W-w+1), indexing='ij')
    iv = iv[(s+th):-(s+th), (s+th):-(s+th)]
    jv = jv[(s+th):-(s+th), (s+th):-(s+th)]
    pos_ref = np.array([iv.reshape(-1), jv.reshape(-1)]).T

    # print(pos_ref.shape)
    # print(pos_mov.shape)

    return pos_ref, pos_mov


cdef inline float compute_low_freq_energy(np.float32_t[:, :] D, int w, int T):
    """ Compute the low frequency energy of a DCT block
        (See algo. 8 of sec. 5.4 in the paper)
    
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

    for i in xrange(w):
        for j in xrange(w):
            if i + j <= T:
                e = e + D[i, j] * D[i, j]
    return e


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_position_pairs(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, \
              np.ndarray[np.int32_t, ndim=2] pos_ref, np.ndarray[np.int32_t, ndim=2] pos_mov, int w, int T, float q):
    """ Select a percentile of block pairs whose difference have the least low-frequency energies
        (See algo. 7 of sec. 5.4 in the paper)

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
        percentile of filtered blocks

    Returns
    -------
    pos_filtered_ref: np.ndarray
        Selected block positions in reference image, of size (M, 2),
        with M = floor(N * q)
    pos_filtered_mov: np.ndarray
        Selected block positions in moving image, of size (M, 2)
    """

    blks_ref = view_as_windows(img_ref, (w, w), step=(1, 1))[pos_ref[:,0], pos_ref[:, 1]] # (N, w, w)
    blks_mov = view_as_windows(img_mov, (w, w), step=(1, 1))[pos_mov[:,0], pos_mov[:, 1]] # (N, w, w)

    cdef int N = pos_ref.shape[0]
    cdef np.ndarray E = np.zeros(N, dtype=np.float32)
    cdef np.float32_t[:] E_view = E
    
    dct_blks_diff = dctn(blks_ref - blks_mov, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)
    dct_blks_diff = dct_blks_diff.astype(np.float32)
    # dct_blks_mov = dctn(blks_mov, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)
    # dct_blks_diff = (dct_blks_ref - dct_blks_mov).astype(np.float32)

    cdef np.float32_t[:, :, :] dct_blks_diff_view = dct_blks_diff
    cdef np.float32_t[:, :] dct_blk_view

    cdef int i
    for i in xrange(N):
        dct_blk_view = dct_blks_diff_view[i]
        E_view[i] = compute_low_freq_energy(dct_blk_view, w, T)

    I = np.argsort(E)[:int(N * q)]

    pos_filtered_ref = pos_ref[I]
    pos_filtered_mov = pos_mov[I]

    return pos_filtered_ref, pos_filtered_mov


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_block_pairs(np.ndarray[np.float32_t, ndim=3] blks_ref, np.ndarray[np.float32_t, ndim=3] blks_mov, int T, float q):
    """ Select a percentile of block pairs whose difference have the least low-frequency energies
        (See algo. 7 of sec. 5.4 in the paper)

    Parameters
    ----------
    blks_ref: np.ndarray
        Blocks in reference image of size (N, w, w)
    blks_mov: np.ndarray
        Blocks in moving image of size (N, w, w)
    T: int
        Threshold for separating the entries for low and high frequency DCT coefficents
    q: float
        percentile of filtered blocks

    Returns
    -------
    pos_filtered_ref: np.ndarray
        Selected blocks in reference image, of size (M, w, w),
        with M = floor(N * q)
    pos_filtered_mov: np.ndarray
        Selected blocks in moving image, of size (M, w, w)
    """

    # assert blks_ref.shape == blks_mov.shape
    cdef int N = blks_mov.shape[0]
    cdef int w = blks_mov.shape[1]

    # blks_ref = view_as_windows(img_ref, (w, w), step=(1, 1))[pos_ref[:,0], pos_ref[:, 1]] # (N, w, w)
    # blks_mov = view_as_windows(img_mov, (w, w), step=(1, 1))[pos_mov[:,0], pos_mov[:, 1]] # (N, w, w)

    cdef np.ndarray E = np.zeros(N, dtype=np.float32)
    cdef np.float32_t[:] E_view = E
    
    dct_blks_ref = dctn(blks_ref, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)
    dct_blks_mov = dctn(blks_mov, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)
    dct_blks_diff = dct_blks_ref - dct_blks_mov

    cdef np.float32_t[:, :, :] dct_blks_diff_view = dct_blks_diff
    cdef np.float32_t[:, :] dct_blk_view

    cdef int i
    for i in xrange(N):
        dct_blk_view = dct_blks_diff_view[i]
        E_view[i] = compute_low_freq_energy(dct_blk_view, w, T)

    I = np.argsort(E)[:int(N * q)]

    blks_filtered_ref = blks_ref[I]
    blks_filtered_mov = blks_mov[I]

    return blks_filtered_ref, blks_filtered_mov


@cython.boundscheck(False)
@cython.wraparound(False)
def partition(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, \
              np.ndarray[np.int32_t, ndim=2] pos_ref, np.ndarray[np.int32_t, ndim=2] pos_mov, int w, int b):
    """ Partition the block pairs into bins according to their mean intensities
        (See algo. 6 of sec. 5.3 in the paper)

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


# Interaction between numpy and native c
# https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html

np.import_array()

cdef extern from "helper.h":
    void find_best_matching(
            float *img_ref, float *img_mov, np.uint16_t *pos_ref, np.uint16_t *pos_mov_init, \
            int H, int W, int N, int w, int th, np.uint16_t *pos_mov_final)

@cython.boundscheck(False)
@cython.wraparound(False)
def find_best_matching_func(
        np.ndarray[float, ndim=2, mode="c"] img_ref not None,
        np.ndarray[float, ndim=2, mode="c"] img_mov not None,
        np.ndarray[np.uint16_t, ndim=2, mode="c"] pos_ref not None,
        np.ndarray[np.uint16_t, ndim=2, mode="c"] pos_mov_init not None,
        int w, int th   
    ):
    """ Wrap C interface into python interface """

    cdef int N = pos_ref.shape[0]
    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] pos_mov = np.zeros((N, 2), dtype=np.uint16)

    find_best_matching(<float*> np.PyArray_DATA(img_ref),
                       <float*> np.PyArray_DATA(img_mov),
                       <np.uint16_t*> np.PyArray_DATA(pos_ref),
                       <np.uint16_t*> np.PyArray_DATA(pos_mov_init),
                       img_ref.shape[0], img_ref.shape[1],
                       N, w, th,
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
    fft = np.fft.fftshift(np.fft.fft2(img_in, norm="forward"))
    fr = fft.real
    fi = fft.imag

    fr = np.pad(fr, ( ((H_obj - H) // 2, ), ((W_obj - W) // 2,) ), 'constant', constant_values=0)
    fi = np.pad(fi, ( ((H_obj - H) // 2, ), ((W_obj - W) // 2,) ), 'constant', constant_values=0)
    
    fft_ups = np.fft.ifftshift(fr + fi * 1j)

    # img_ups = np.fft.ifft2(fft_ups, norm="forward").real
    img_ups = ifft2(fft_ups, norm="forward", workers=8).real

    img_ups = np.clip(img_ups, 0, 1) * 255
    img_ups = img_ups.astype(np.float16)

    return img_ups
    

@cython.boundscheck(False)
@cython.wraparound(False)
def subpixel_match(img_ref, img_mov, pos_ref, pos_mov_init, w, th, num_iter=2):
    """ Compute the blocks in the moving image that match the blocks in the reference image
        by matching their surrounding rings in subpixelic precision
        (See algo. 5 of sec. 5.2 in the paper)

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
    num_iter: int
        Number of iterations for iterative subpixel matching. The matching precision is doubled
        at each iteration (e.g. 1/2px -> 1/4px -> 1/8px etc.)

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

    sz_patch = w + 2 * th

    sz_patch_iter = sz_patch
    w_up = w
    th_up = th

    # pos_ref_up = pos_ref.copy()
    # pos_mov_up = pos_mov_init.copy()

    # TODO: add to IPOL paper
    # remove blocks within 1 pixel from the border
    cdef int margin = 2
    valid_mask = (pos_mov_init[:, 0] >= th + margin) & (pos_mov_init[:, 0] < H-sz_patch+1 - th - margin) & \
                     (pos_mov_init[:, 1] >= th + margin) & (pos_mov_init[:, 1] < W-sz_patch+1 - th - margin)
    pos_ref = pos_ref[valid_mask]
    
    pos_ref_up = pos_ref.copy()
    pos_mov_up = pos_mov_init[valid_mask]


    for iter in range(num_iter):
        # 1. upsampling
        ups_factor = 2 ** (iter+1)
        img_ups_ref = upsample_image(img_ref, ups_factor)
        img_ups_mov = upsample_image(img_mov, ups_factor)
        
        pos_ref_up = 2 * pos_ref_up
        pos_mov_up = 2 * pos_mov_up

        sz_patch_iter = 2 * sz_patch_iter
        w_up = 2 * w_up
        th_up = 2 * th_up

        # C function requires c-contiguous array
        img_ups_ref = np.ascontiguousarray(img_ups_ref, dtype=np.float32)
        img_ups_mov = np.ascontiguousarray(img_ups_mov, dtype=np.float32)
        pos_ref_up = np.ascontiguousarray(pos_ref_up, dtype=np.uint16)
        pos_mov_up = np.ascontiguousarray(pos_mov_up, dtype=np.uint16)

        pos_mov_up = find_best_matching_func(img_ups_ref, img_ups_mov, pos_ref_up, pos_mov_up, w_up, th_up)

    sz_patch_up = sz_patch * (2 ** num_iter)
    img_split_mov = view_as_windows(img_ups_mov, (w_up, w_up), step=(1, 1)) #.reshape(-1, w_up, w_up)


    # pos_mov_final = pos_mov_init


    # print(img_split_mov.shape)
    # print("i max", pos_mov_up[:, 0].max())
    # print("j max", pos_mov_up[:, 1].max())

    blocks_mov = img_split_mov[pos_mov_up[:, 0], pos_mov_up[:, 1]] # (N, w * 2**num_iter, w * 2**num_iter,)
    blocks_mov = view_as_blocks(blocks_mov, (1, 2**(num_iter), 2**(num_iter))) # (N, w, w, 1, 2**num_iter, 2**num_iter)

    blocks_mov = blocks_mov[:,:, :,0,0,0]

    # blocks_mov = blocks_mov[:, th:(w+th), th:(w+th)]

    # DEBUG use
    # pos_mov_final = pos_mov_up / 2**(num_iter)

    blocks_ref = view_as_windows(img_ref, (w, w), step=(1, 1))[pos_ref[:, 0], pos_ref[:, 1]] # (N, w, w)
    
    # DEBUG use
    # return pos_ref, pos_mov_final, blocks_ref.astype(np.float32), blocks_mov.astype(np.float32)

    return blocks_ref.astype(np.float32), blocks_mov.astype(np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_variance_from_pairs(blks_ref, blks_mov, T):
    """ Compute noise variance from block pairs
        (See Algo. ? of Sec. ? in the paper)
    
    Parameters
    ----------
    blks_ref: ndarray
        Blocks in reference image of size (N, w, w)
    blks_mov: ndarray
        Blocks in moving image of size (N, w, w)
    T: int
        Threshold for separating the entries for low and high frequency DCT coefficents

    Returns
    -------
    variance: float
        Noise variance
    """

    assert blks_ref.shape == blks_mov.shape
    cdef int N = blks_ref.shape[0]
    cdef int w = blks_ref.shape[1]

    # blks_diff = blks_mov - blks_ref
    dct_blks_ref = dctn(blks_mov - blks_ref, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)

    VH = []
    cdef int i, j
    for i in xrange(w):
        for j in xrange(w):
            if i + j > T:
                VH.append(np.mean(dct_blks_ref[:, i, j] ** 2))
    
    return np.median( np.array(VH) ) / 2
