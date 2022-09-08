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

from scipy.fft import dct, dctn, fft2, ifft2
from scipy.signal import convolve2d
from scipy import interpolate

import os


# For debug use
# from inspect import currentframe

########################################

ctypedef fused T_t:
    np.int32_t
    float

ctypedef fused Tu_t:
    np.int64_t
    double

# ctypedef np.int32_t T_t
# T = np.int32

cdef extern from "limits.h":
    cdef int INT32_MAX
    cdef unsigned int UINT32_MAX
    cdef long INT64_MAX
    cdef unsigned long UINT64_MAX

cdef extern from "float.h":
    cdef double DBL_MAX

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
    
    Return
    ----------
    img_dir: np.ndarray
        Image of uint8 values in [0, k-1]. Each int value indicates a gradient direction.

    """

    img_dx = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 1, 0, 3)
    img_dy = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 0, 1, 3)
    img_angle = np.arctan2(img_dy, img_dx) # [-pi, pi)
    # img_dir = np.floor(img_angle / 2 / np.pi * k)
    # img_dir[img_dir < 0] += k

    img_dir = np.round(img_angle / 2 / np.pi * k)

    img_dir = img_dir.astype(np.int8) % k

    return img_dir


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

    knl = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)

    img_blur = convolve2d(img, knl, mode="same", boundary="symm")

    return img_blur


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
    if T_t is int:
        T = np.int64
    else:
        T = np.float64

    cdef np.ndarray img_int = np.zeros((H, W), dtype=T)
    img_int[0] = np.cumsum(img[0])

    cdef T_t s
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

    sum_of_blocks = np.cumsum(img, axis=1, dtype=np.float64)

    sum_of_blocks[:, w:] = sum_of_blocks[:, w:] - sum_of_blocks[:, :-w]
    sum_of_blocks = sum_of_blocks[:, w-1 : ]

    sum_of_blocks = np.cumsum(sum_of_blocks, axis=0, dtype=np.float64)
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


# @cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def pixel_match(np.ndarray[T_t, ndim=2] img_ref, np.ndarray[T_t, ndim=2] img_mov, int w, int th, int s):
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

    # DEBUG
    # cdef np.ndarray img_blur_ref = img_ref.copy().astype(np.int64)
    # cdef np.ndarray img_blur_mov = img_mov.copy().astype(np.int64)

    cdef np.ndarray img_blur_ref = img_blur(img_ref)
    cdef np.ndarray img_blur_mov = img_blur(img_mov)


    # cdef np.ndarray offsets = np.zeros((2*s+1, 2*s+1), dtype=np.int64)


    cdef np.ndarray img_diff_offsets = np.zeros((2*s+1, 2*s+1, H-2*s, W-2*s), dtype=np.float64)
    
    cdef double[:, :, :, :] img_diff_offsets_view = img_diff_offsets
    cdef double[:, :] img_blur_ref_view = img_blur_ref
    cdef double[:, :] img_blur_mov_view = img_blur_mov
    
    # cdef np.int32_t[:, :] img_diff_view

    img_blur_ref_cropped = img_blur_ref[s:(H-s), s:(W-s)]

    cdef int off_i, off_j, i, j

    # TODO: parallelism
    # for off_i in prange(0, 2*s+1, nogil=True, num_threads=8):
    for off_i in range(0, 2*s+1):
        for off_j in xrange(0, 2*s+1):
            # img_blur_mov_cropped = img_blur_mov[off_i:(H-2*s+off_i), off_j:(W-2*s+off_j)]
            # img_diff_offsets[off_i, off_j] = img_blur_ref_cropped - img_blur_mov_cropped
            # img_diff_offsets[off_i, off_j] = img_diff_offsets[off_i, off_j] * img_diff_offsets[off_i, off_j]


            # img_diff_view = img_diff_offsets_view[off_i, off_j]
            for i in xrange(0, H-2*s):
                for j in xrange(0, W-2*s):
                    # TODO: update SSD metric in the paper
                    img_diff_offsets_view[off_i, off_j, i, j] = img_blur_ref_view[i+s, j+s] - img_blur_mov_view[i+off_i, j+off_j]
                    img_diff_offsets_view[off_i, off_j, i, j] = img_diff_offsets_view[off_i, off_j, i, j] * img_diff_offsets_view[off_i, off_j, i, j]

    cdef int outer_sz = 2 * th + w
    cdef np.ndarray pos_ref = np.zeros(( (H-2*s-outer_sz+1) * (W-2*s-outer_sz+1), 2), dtype=np.int32)
    cdef np.ndarray pos_mov = np.zeros(( (H-2*s-outer_sz+1) * (W-2*s-outer_sz+1), 2), dtype=np.int32)
    
    cdef np.ndarray cost_of_offsets = np.zeros((2*s+1, 2*s+1, H-2*s-outer_sz+1, W-2*s-outer_sz+1), dtype=np.float64)
    # cdef np.ndarray sum_of_outer_blks, sum_of_inner_blks
    
    for off_i in xrange(2*s+1):
        for off_j in xrange(2*s+1):
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

    # TODO: parallelism
    # openmp.omp_set_dynamic(8)
    
    for i in range(H_img_blk):
        for j in range(W_img_blk):
            # cost_best = INT32_MAX 
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

            # nb_pos = nb_pos + 1


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

    # return L

    return bins_pos_ref, bins_pos_mov


# Interaction between numpy and native c
# https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html

np.import_array()

cdef extern from "helper.h":
    void find_best_matching(
            float *img_ref, float *img_mov, np.uint16_t *pos_ref, np.uint16_t *pos_mov_init, \
            int H, int W, int N, int w, int th,int ups_factor, np.uint16_t *pos_mov_final)

    void find_best_matching_one_shot(
            float *img_ref, float *img_ups_mov, np.uint16_t *pos_ref, np.uint16_t *pos_mov_init, \
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
    find_best_matching_one_shot(<float*> np.PyArray_DATA(img_ref),
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

    # img_ups = np.fft.ifft2(fft_ups, norm="forward").real
    img_ups = ifft2(fft_ups, norm="forward", workers=os.cpu_count()).real

    # img_ups = np.clip(img_ups, 0, 1) * 255
    img_ups = img_ups * 255.0
    img_ups = img_ups.astype(np.float32)

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

    # sz_patch_iter = sz_patch
    w_up = w
    th_up = th

    # pos_ref_up = pos_ref.copy()
    # pos_mov_up = pos_mov_init.copy()

    # TODO: add to IPOL paper
    # remove blocks within 1 pixel from the border
    cdef int margin = 2
    valid_mask = (pos_mov_init[:, 0] >= th + margin) & (pos_mov_init[:, 0] < H - w + 1 - th - margin) & \
                     (pos_mov_init[:, 1] >= th + margin) & (pos_mov_init[:, 1] < W - w + 1 - th - margin)
    pos_ref = pos_ref[valid_mask]
    
    # pos_ref_up = pos_ref.copy()
    pos_mov_up = pos_mov_init[valid_mask]

    # Start matching
    ups_factor = 2 ** num_iter
    # img_ups_ref = upsample_image(img_ref, ups_factor)
    img_ups_mov = upsample_image(img_mov, ups_factor)

    # print(f"img_ups_mov\n", img_ups_mov)
    
    # pos_ref_up = pos_ref_up * ups_factor
    pos_mov_up = pos_mov_up * ups_factor

    sz_patch_up = sz_patch * ups_factor
    w_up = w * ups_factor
    th_up = th * ups_factor

    # C function requires c-contiguous array
    # if (not img_ups_ref.flags["C_CONTIGUOUS"]) or (img_ups_ref.dtype != np.float32):
    #     img_ups_ref = np.ascontiguousarray(img_ups_ref, dtype=np.float32)
    if (not img_ref.flags["C_CONTIGUOUS"]) or (img_ref.dtype != np.float32):
        img_ref = np.ascontiguousarray(img_ref, dtype=np.float32)
    if (not img_ups_mov.flags["C_CONTIGUOUS"]) or (img_ups_mov.dtype != np.float32):
        img_ups_mov = np.ascontiguousarray(img_ups_mov, dtype=np.float32)
    if (not pos_ref.flags["C_CONTIGUOUS"]) or (pos_ref.dtype != np.uint16):
        pos_ref = np.ascontiguousarray(pos_ref, dtype=np.uint16)
    if (not pos_mov_up.flags["C_CONTIGUOUS"]) or (pos_mov_up.dtype != np.uint16):
        pos_mov_up = np.ascontiguousarray(pos_mov_up, dtype=np.uint16)

    # from datetime import datetime
    # print(f"find_best_matching_func begins at {datetime.now()}")
    pos_mov_up = find_best_matching_func(img_ref, img_ups_mov, pos_ref, pos_mov_up, w, th, ups_factor)
    
    # print(f"find_best_matching_func ends at {datetime.now()}")




    # for iter in range(num_iter):
    #     # 1. upsampling
    #     ups_factor = 2 ** (iter+1)
    #     img_ups_ref = upsample_image(img_ref, ups_factor)
    #     img_ups_mov = upsample_image(img_mov, ups_factor)

    #     # print(f"img_ups_mov\n", img_ups_mov)
        
    #     pos_ref_up = 2 * pos_ref_up
    #     pos_mov_up = 2 * pos_mov_up

    #     sz_patch_iter = 2 * sz_patch_iter
    #     w_up = 2 * w_up
    #     th_up = 2 * th_up

    #     # C function requires c-contiguous array
    #     if (not img_ups_ref.flags["C_CONTIGUOUS"]) or (img_ups_ref.dtype != np.float32):
    #         img_ups_ref = np.ascontiguousarray(img_ups_ref, dtype=np.float32)
    #     if (not img_ups_mov.flags["C_CONTIGUOUS"]) or (img_ups_mov.dtype != np.float32):
    #         img_ups_mov = np.ascontiguousarray(img_ups_mov, dtype=np.float32)
    #     if (not pos_ref_up.flags["C_CONTIGUOUS"]) or (pos_ref_up.dtype != np.uint16):
    #         pos_ref_up = np.ascontiguousarray(pos_ref_up, dtype=np.uint16)
    #     if (not pos_mov_up.flags["C_CONTIGUOUS"]) or (pos_mov_up.dtype != np.uint16):
    #         pos_mov_up = np.ascontiguousarray(pos_mov_up, dtype=np.uint16)

    #     from datetime import datetime
    #     print(f"find_best_matching_func begins at {datetime.now()}")
    #     pos_mov_up = find_best_matching_func(img_ups_ref, img_ups_mov, pos_ref_up, pos_mov_up, w_up, th_up, ups_factor)
        
    #     print(f"find_best_matching_func ends at {datetime.now()}")

    # sz_patch_up = sz_patch * ups_factor

    img_split_mov = view_as_windows(img_ups_mov, (w_up, w_up), step=(1, 1)) #.reshape(-1, w_up, w_up)
    blocks_mov = img_split_mov[pos_mov_up[:, 0], pos_mov_up[:, 1]] # (N, w * 2**num_iter, w * 2**num_iter,)
    blocks_mov = view_as_blocks(blocks_mov, (1, 2**(num_iter), 2**(num_iter))) # (N, w, w, 1, 2**num_iter, 2**num_iter)
    blocks_mov = blocks_mov[:,:,:,0,0,0]



    # faster version
    img_split_ref = view_as_windows(img_ref, (w, w), step=(1, 1))
    blocks_ref = img_split_ref[pos_ref[:, 0], pos_ref[:, 1]] # (N, w, w)


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


from scipy.fft import dctn
from skimage.util.shape import view_as_blocks


@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_variance_from_differences(
    blks_ref: np.ndarray, blks_mov: np.ndarray,
    T: int, q: float, scale:int):
    """
    Parameters
    ----------
    blks_ref: blocks in reference frame, in shape (N, w, w)
    blks_mov: blocks in moving frame, in shape (N, w, w)
    T: frequency separator
    q: percentile of selected blocks for estimation
    scale: (optional) subsample the blocks at the specific scale

    Return
    ------
    variance: noise variance from the blocks of differences

    """

    blks_diff = blks_mov - blks_ref

    dct_blks = dctn(blks_diff, axes=(-1,-2), norm='ortho', workers=8) # (N, w, w)

    # if scale is not 0, subsample the dct blocks
    N, W, _ = dct_blks.shape

    w = W
    if scale > 0:
        factor = 2 ** scale
        assert W % factor == 0, "The block size must be multiple of the subsample factor"

        w = W // factor
        blks = view_as_blocks(dct_blks, (1, factor, factor)).squeeze() # (N, w, w, factor, factor)
        blks = np.transpose(blks, (0, 3, 4, 1, 2)) # (N, factor, factor, w, w)
        dct_blks = blks.reshape(-1, w, w)

    
    # Low-frequency energy
    cdef np.ndarray E = np.zeros(N, dtype=np.float32)
    cdef np.float32_t[:] E_view = E

    cdef np.float32_t[:, :, :] dct_blks_view = dct_blks
    cdef np.float32_t[:, :] dct_blk_view

    cdef int i
    for i in xrange(N):
        dct_blk_view = dct_blks_view[i]
        E_view[i] = compute_low_freq_energy(dct_blk_view, w, T)

    I = np.argsort(E)[:int(N * q)]

    dct_blks_good = dct_blks[I] # (n, w, w)


    VH = []
    for i in xrange(w):
        for j in xrange(w):
            if i + j > T:
                VH.append(np.mean(dct_blks_good[:, i, j] ** 2))
    
    var = np.median( np.array(VH) ) / 2

    intensity = (blks_mov[I].mean() + blks_ref[I].mean()) / 2

    return intensity, var







