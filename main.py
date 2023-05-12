
import time
from src.estimate import estimate_noise_curve

import src.utils as utils
import numpy as np
import os

import argparse


parser = argparse.ArgumentParser(description='Video Signal-Dependent Noise Estimation via Inter-frame Prediction. '
                                 '(c) 2022 Yanhao Li. Under license GNU AGPL.')

parser.add_argument('im_0', type=str,
                    help='First frame filename')
parser.add_argument('im_1', type=str,
                    help='Second frame filename')
parser.add_argument('-bins', type=int, default=16,
                    help='Number of bins')
parser.add_argument('-q', type=float, default=0.01,
                    help='Quantile of block pairs')
parser.add_argument('-s', type=int, default=5,
                    help='Search radius of patch matching')
parser.add_argument('-w', type=int, default=8,
                    help='Block size')
parser.add_argument('-T', type=int, default=-1,
                    help='Frequency separator')
parser.add_argument('-th', type=int, default=3,
                    help='Thickness of ring for patch matching')
parser.add_argument('-g', default=False,
                    help='Whether the input images are in grayscale', action='store_true')
parser.add_argument('-add_noise', default=False,
                    help='True for adding simulated noise',
                    action='store_true')
parser.add_argument('-noise_a', type=float, default=0.2,
                    help='Noise model parameter: a')
parser.add_argument('-noise_b', type=float, default=0.2,
                    help='Noise model parameter: b')
args = parser.parse_args()


def save_to_txt(intensities, variances, save_fname):
    """ Save to a txt file of size (bins, channels * 2)
    intensities: of size (channels, bins)
    variances: of size (channels, bins)
    save_fname: the txt filename
    """
    assert intensities.shape == variances.shape
    channels, bins = intensities.shape
    out_data = np.zeros((bins, channels * 2))
    for c in range(channels):
        out_data[:, c*2] = intensities[c, :]
        out_data[:, c*2+1] = variances[c, :]
    
    len = int(np.log10(np.abs(out_data).max())) + 4
    np.savetxt(save_fname, out_data, fmt=f'%{len}.3f')


def main():
    print("Parameters:")
    print(args)
    print()

    if args.T == -1:
        args.T = args.w + 1

    supported_ext = [".tif", ".tiff", ".dng"]

    # verify the two images are in the same extension
    _, extension_0 = os.path.splitext(args.im_0)
    _, extension_1 = os.path.splitext(args.im_1)
    assert extension_0 in supported_ext, \
        f"Only `.tif`, `.tiff` and `.dng` formats are support, but `{extension_0}` was found."
    assert extension_0 == extension_1, \
        f"The two input images must be in the same format, but `{extension_0}` and `{extension_1}` were got."
    
    img_0 = utils.read_img(args.im_0, grayscale=args.g)
    img_1 = utils.read_img(args.im_1, grayscale=args.g)

    if args.add_noise == True:
        img_0 = utils.add_noise(img_0, args.noise_a, args.noise_b)
        img_1 = utils.add_noise(img_1, args.noise_a, args.noise_b)

    if img_0.shape != img_1.shape: 
        print("Error: The two input images should have the same size and the same channel")
        quit()

    if args.T > 2 * args.w - 3:
        print("Error: Frequency separator T and block size w should satisfy T<=2*w-3")
        quit()
        

    start = time.time()

    img_0 = img_0.astype(np.float32)
    img_1 = img_1.astype(np.float32)
    intensities, variances = estimate_noise_curve(img_0, img_1, w=args.w, T=args.T, th=args.th, q=args.q, bins=args.bins, s=args.s)

    save_to_txt(intensities, variances, f"curve_s0.txt" )
    utils.plot_noise_curve(intensities, variances, fname=f"curve_s0.png")

    print(f"time spent: {time.time() - start} s")

if __name__ == "__main__":
    main()
