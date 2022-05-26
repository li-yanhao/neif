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

import numpy as np
cimport numpy as np

import cv2
from skimage.util import view_as_windows

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
    img : np.ndarray
        One channel input image
    k : int
        Number of areas to divide in 360 degrees
    
    Return
    ----------
    img_dir: image of int values in [0, k-1]. Each int value indicates a gradient direction.

    """

    img_dx = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 1, 0, 3)
    img_dy = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 0, 1, 3)
    img_angle = np.arctan2(img_dy, img_dx) # [-pi, pi)
    img_dir = np.floor(img_angle / 2 / np.pi * k)

    img_dir[img_dir < 0] += k
    img_dir = img_dir.astype(np.uint8)

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


# @cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.uint64_t compute_cost(np.uint64_t[:, :] img_int, int pi, int pj, int w):
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
    # cdef int pi = p[0]
    # cdef int pj = p[1]
    # cdef np.uint64_t cost = img_int[pi + w - 1, pj + w - 1]
    # if pi > 0:
    #     cost = cost - img_int[pi - 1, pj + w -1]
    # if pj > 0:
    #     cost = cost - img_int[pi + w - 1, pj - 1]
    # if pi > 0 and pj > 0:
    #     cost = cost + img_int[pi - 1, pj - 1]
    
    cdef np.uint64_t cost
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
cdef inline int dist(np.uint8_t x, np.uint8_t y, int k):
    cdef np.uint8_t d0, d1
    if x > y: d0 = x - y
    else: d0 = y - x
    d1 = k - d0
    if d0 < d1: return d0
    else: return d1


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
    pos_ref: numpy.ndarray
        2D positions of blocks in img_ref, of size (N, 2)
    pos_mov: numpy.ndarray
        2D positions of blocks in img_mov, of size (N, 2)
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
    cdef np.ndarray img_integral_offsets = np.zeros((2*s+1, 2*s+1, H-2*s, W-2*s), dtype=np.uint64)
    
    cdef np.uint64_t[:, :, :, :] img_integral_offsets_view = img_integral_offsets
    cdef np.uint8_t[:, :] img_dir_ref_view = img_dir_ref
    cdef np.uint8_t[:, :] img_dir_mov_view = img_dir_mov
    
    cdef np.uint64_t[:, :] img_diff_view
    cdef int off_i, off_j, i, j
    for off_i in xrange(2*s+1):
        for off_j in xrange(2*s+1):
            img_diff_view = img_integral_offsets_view[off_i, off_j]
            for i in xrange(H-2*s):
                for j in xrange(W-2*s):
                    img_diff_view[i, j] = dist(img_dir_ref_view[i+off_i, j+off_j], img_dir_mov_view[i+off_i, j+off_j], k) # update k in paper

    cdef int outer_sz = 2 * th + w
    cdef np.ndarray pos_ref = np.zeros(( (H-2*s-outer_sz+1) * (W-2*s-outer_sz+1), 2), dtype=np.int32)
    
    cdef np.uint64_t cost_best, cost_outer, cost_inner
    cdef int off_i_best, off_j_best
    for i in range(H-2*s-outer_sz+1):
        for j in range(W-2*s-outer_sz+1):
            cost_best = UINT32_MAX
            off_i_best = 0
            off_j_best = 0
            for off_i in range(2*s+1):
                for off_j in range(2*s+1):
                    cost_outer = compute_cost(img_integral_offsets_view[off_i, off_j], i, j, outer_sz)
                    cost_inner = compute_cost(img_integral_offsets_view[off_i, off_j], i+th, j+th, w)


    pass


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
    
    # arr_2d = np.arange((H-s*2-w+1) * (W-s*2-w+1)).reshape(H-s*2-w+1, W-s*2-w+1)
    iv, jv = np.meshgrid(np.arange(H-s*2-sz_patch+1), np.arange(W-s*2-sz_patch+1), indexing='ij')

    arr_2d = (iv * (W-sz_patch+1) + jv)
    match_indices_final = arr_2d.reshape(-1) + match_indices // (2*s+1) * (W-sz_patch+1) + match_indices % (2*s+1)

    img_mov_blocks = view_as_windows(img_mov, (sz_patch, sz_patch), step=(1, 1)).reshape(-1, sz_patch, sz_patch)
    img_mov_blocks_best = img_mov_blocks[match_indices_final]
    
    img_ref_cropped_blocks = view_as_windows(img_ref[s:(H-s), s:(W-s)], (sz_patch, sz_patch), step=(1, 1)).reshape(-1, sz_patch, sz_patch)

    print("patch match done")

    return img_ref_cropped_blocks[..., th:(w+th), th:(w+th)], img_mov_blocks_best[..., th:(w+th), th:(w+th)]


def subpixel_match():
    """ xxx
        (See algo. 5 of sec. 5.2 in the paper)
    """
    pass


def filter_block_pairs():
    """
        (See algo. 6 of sec. 5.4 in the paper)
    """
    pass