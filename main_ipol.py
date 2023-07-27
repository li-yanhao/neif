import time
import os
import subprocess
import argparse

import numpy as np
import magic
import skimage.io as iio

from src.estimate import estimate_noise_curve_v2
import src.utils as utils



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


def restore_file_ext(fname):
    metadata = magic.from_file(fname, mime=True)
    ext = metadata.split("/")[-1]
    if not fname.endswith(ext):
        fname_root, _ = os.path.splitext(fname)
        new_fname = fname_root + "." + ext
        new_fname = fname + "." + ext
        command = f"mv {fname} {new_fname}"
        subprocess.run(command, shell=True, capture_output=True, text=True)
        return new_fname
    else:
        return fname


def save_img(path, img):
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
    img = img.astype(np.uint8)
    iio.imsave(path, img.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description='Video Signal-Dependent Noise Estimation via Inter-frame Prediction.\n'
                                    'This script is for IPOL demo use.\n'
                                    '(c) 2022 Yanhao Li. Under license GNU AGPL.')

    parser.add_argument('im_0', type=str,
                        help='First frame filename')
    parser.add_argument('im_1', type=str,
                        help='Second frame filename')
    parser.add_argument('-grayscale', type=str,
                        default="false", choices=["true", "false"],
                        help='Whether the input images are in grayscale (default: False)')
    parser.add_argument('-bins', type=int,
                        help='Number of bins')
    parser.add_argument('-q', type=float,
                        help='Quantile of block pairs')
    parser.add_argument('-s', type=int,
                        help='Search radius of patch matching')
    parser.add_argument('-w', type=int,
                        help='Block size')
    parser.add_argument('-th', type=int,
                        help='Thickness of ring for patch matching (default: 3)')
    parser.add_argument('-add_noise', type=str,
                        default="false", choices=["true", "false"],
                        help='True for adding simulated noise')
    parser.add_argument('-noise_a', type=float,
                        help='Noise model parameter: a')
    parser.add_argument('-noise_b', type=float, default=0.2,
                        help='Noise model parameter: b')
    parser.add_argument('-f_us', type=int,
                        help='Upsampling scale for subpixel matching')
    args = parser.parse_args()

    print("Parameters:")
    print(args)
    print()

    # Convert IPOL parameters
    T = args.w + 1
    if args.grayscale == "true":
        grayscale = True
    else:
        grayscale = False
    if args.add_noise == "true":
        add_noise = True
    else:
        add_noise = False

    supported_ext = [".tif", ".tiff", ".dng", ".png", ".jpg", ".jpeg"]

    fname_0 = restore_file_ext(args.im_0)
    fname_1 = restore_file_ext(args.im_1)

    # verify the two images are in the same extension
    _, ext_0 = os.path.splitext(fname_0)
    _, ext_1 = os.path.splitext(fname_1)

    assert ext_0 in supported_ext, \
        f"Only `.tif`, `.tiff`, `.dng`, `.png`, `.jpg` and `.jpeg` formats are support, but `{ext_0}` was found."
    assert ext_0 == ext_1, \
        f"The two input images must be in the same format, but `{ext_0}` and `{ext_1}` were found."
    
    if ext_0 == ".tiff" or ext_0 == ".tif" or ext_0 == ".dng":
        is_raw = True
    else:
        is_raw = False


    img_0 = utils.read_img(fname_0, grayscale=grayscale)
    img_1 = utils.read_img(fname_1, grayscale=grayscale)

    if add_noise:
        img_0 = utils.add_noise(img_0, args.noise_a, args.noise_b)
        img_1 = utils.add_noise(img_1, args.noise_a, args.noise_b)

    # Save image for visualization in IPOL demo
    save_img("noisy_0.png", img_0)
    save_img("noisy_1.png", img_1)

    if img_0.shape != img_1.shape: 
        print("Error: The two input images should have the same size and the same channel")
        quit()

    start = time.time()
    
    img_0 = img_0.astype(np.float32)
    img_1 = img_1.astype(np.float32)
    intensities, variances = estimate_noise_curve_v2(img_0, img_1, w=args.w, T=T, th=args.th, q=args.q, bins=args.bins, s=args.s, f_us=args.f_us, is_raw=is_raw)

    num_scale = intensities.shape[0]
    for scale in range(num_scale):
        save_to_txt(intensities[scale], variances[scale], f"curve_s{scale}.txt" )
        utils.plot_noise_curve(intensities[scale], variances[scale], fname=f"curve_s{scale}.png")

    print(f"time spent: {time.time() - start} s")

if __name__ == "__main__":
    main()
