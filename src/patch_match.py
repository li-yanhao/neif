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
from skimage.util import view_as_windows


def img_direction(img, k):
    """ Divide 360 degrees into k ranges, compute the gradient directions of each pixel and 
        map with the directions with an integer in [0, k-1].
    Parameters
    ----------
    img : np.ndarray
        One channel input image
    k : int
        Number of areas to divide in 360 degrees
    
    Return
    ----------
    image of int values in [0, k-1]. Each int value indicates a gradient direction.

    """

    img_dx = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 1, 0, 3)
    img_dy = cv2.Sobel(img.astype(np.uint16), cv2.CV_32F, 0, 1, 3)
    out = np.arctan((img_dy) / (img_dx+0.0001))
    out = np.round((out + np.pi / 2) / (np.pi / k * 2))

    out[img_dx < 0] += k / 2
    out = out.astype(np.int8) % k
    # print(out)
    return out


def patch_match(img_0, img_1, w, th, s, num_div):
    """ Dividing two images into wxw blocks. For each block in img_0 search the matched block in img_1 within a search range.
    
    Parameters
    ----------
    img_0 : numpy.ndarray
        Raw image at t, in size (H, W)
    img_1 : numpy.ndarray
        Raw image at t+1, in size (H, W)
    w : int
        Block size
    s : int
        Half of search range for patch matching. Note that the seach offset is within [-s, +s]
    th : int
        Thickness of the ring bounding the patch
    num_div : int
        Number of divided areas of 360 degrees
    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        [0]: Raw blocks in img_0, of size (N, w, w), N = (H - w + 1 - 2 * search_range) * (W - w + 1 - 2 * search_range)
        [1]: Raw blocks in img_1 that match the blocks in img_0, of size (N, w, w)
    """


    H, W = img_0.shape

    img_0_dir = img_direction(img_0, num_div)
    img_1_dir = img_direction(img_1, num_div)

    img_0_dir_cropped = img_0_dir[s:(H-s), s:(W-s)]

    img_dir_diff_cdds = np.zeros(((2*s+1)**2, (H-s*2), (W-s*2)), dtype=np.uint8)
    idx = 0
    for i in range(-s, s+1):
        for j in range(-s, s+1):
            img_1_dir_cropped = img_1_dir[(s+i):(H-s+i), (s+j):(W-s+j)]
            img_dir_diff_cropped = np.abs(img_0_dir_cropped - img_1_dir_cropped)
            img_dir_diff_cropped = np.minimum(img_dir_diff_cropped, num_div - img_dir_diff_cropped)
            
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

    img_1_blocks = view_as_windows(img_1, (sz_patch, sz_patch), step=(1, 1)).reshape(-1, sz_patch, sz_patch)
    img_1_blocks_best = img_1_blocks[match_indices_final]
    
    img_0_cropped_blocks = view_as_windows(img_0[s:(H-s), s:(W-s)], (sz_patch, sz_patch), step=(1, 1)).reshape(-1, sz_patch, sz_patch)

    print("patch match done")

    return img_0_cropped_blocks[..., th:(w+th), th:(w+th)], img_1_blocks_best[..., th:(w+th), th:(w+th)]