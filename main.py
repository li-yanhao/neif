
import time
from src.estimate import estimate_noise_curve
from src.estimate import estimate_noise_curve_subpixel

import src.utils as utils
import cv2
import numpy as np
import os

import argparse


parser = argparse.ArgumentParser(description='Noise Estimation')

parser.add_argument('im_0', type=str,
                    help='First frame filename')
parser.add_argument('im_1', type=str,
                    help='Second frame filename')
parser.add_argument('-bins', type=int, default=16,
                    help='Number of bins')
parser.add_argument('-quantile', type=float, default=0.01,
                    help='Quantile of block pairs')
parser.add_argument('-search_range', type=int, default=5,
                    help='Search range of patch matching')
parser.add_argument('-w', type=int, default=8,
                    help='Block size')
parser.add_argument('-T', type=int, default=-1,
                    help='Frequency separator')
parser.add_argument('-th', type=int, default=3,
                    help='Thickness of ring for patch matching')
parser.add_argument('-grayscale', default=False,
                    help='Whether the input image is in grayscale'
                    'noise estimation', action='store_true')
parser.add_argument('-add_noise', default=False,
                    help='True for adding simulated noise',
                    action='store_true')
parser.add_argument('-noise_a', type=float, default=0.2,
                    help='Noise model parameter: a')
parser.add_argument('-noise_b', type=float, default=0.2,
                    help='Noise model parameter: b')
parser.add_argument('-multiscale', type=int, default=-1,
                    help='Number of scales for downscaling. -1 for automatic selection of scales.')
parser.add_argument('-subpx_order', type=int, default=0,
                    help='Upsampling scale for subpixel matching')
args = parser.parse_args()


def save_to_txt(intensities, variances, scale):
    """ Save to a txt file of size (bins, channels * 2)
    intensities: (channels, bins)
    variances: (channels, bins)
    
    """
    assert intensities.shape == variances.shape
    channels, bins = intensities.shape
    out_data = np.zeros((bins, channels * 2))
    for c in range(channels):
        out_data[:, c*2] = intensities[c, :]
        out_data[:, c*2+1] = variances[c, :]
    
    len = int(np.log10(np.abs(out_data).max())) + 4
    np.savetxt(f"output_s{scale}.txt", out_data, fmt=f'%{len}.3f')


def main_old():
    print("Parameters:")
    print(args)
    print()
    

    img_0 = utils.read_img(args.im_0, grayscale=args.grayscale)
    img_1 = utils.read_img(args.im_1, grayscale=args.grayscale)

    if args.add_noise == True:
        img_0_noisy = utils.add_noise(img_0, args.noise_a, args.noise_b)
        img_1_noisy = utils.add_noise(img_1, args.noise_a, args.noise_b)

        noise_0 = img_0_noisy - img_0
        noise_1 = img_1_noisy - img_1

        # scale noise for visualization
        max_val = np.max((np.max(noise_0), np.max(noise_1)))
        noise_0 = np.uint8(np.abs(noise_0) / max_val * 255)
        noise_1 = np.uint8(np.abs(noise_1) / max_val * 255)
        print(noise_0.shape)
        noise_0 = np.transpose(noise_0, (1, 2, 0))
        noise_1 = np.transpose(noise_1, (1, 2, 0))
        cv2.imwrite(f"noise_0.png", noise_0)
        cv2.imwrite(f"noise_1.png", noise_1)

        img_0 = img_0_noisy
        img_1 = img_1_noisy
        

    if img_0.shape != img_1.shape: 
        print("Error: The two input images should have the same size and the same channel")
        quit()

    if args.T > 2 * args.w - 3:
        print("Error: Frequency separator T and block size w should satisfy T<=2*w-3")
        quit()
        

    start = time.time()
    
    # args.T = args.w + 1

    scale = 0

    # save image for result visualization
    img_0_v = np.transpose(img_0, (1, 2, 0))
    img_1_v = np.transpose(img_1, (1, 2, 0))
    cv2.imwrite(f"noisy_0.png", img_0_v.astype(np.uint8))
    cv2.imwrite(f"noisy_1.png", img_1_v.astype(np.uint8))

    print("###### Output ###### \n")

    while scale <= args.multiscale:

        img_0 = img_0.astype(np.float32)
        img_1 = img_1.astype(np.float32)

        factor = 2**scale

        if args.T == -1:
            args.T = args.w + 1

        intensities, variances = estimate_noise_curve(img_0, img_1, w=args.w, T=args.T, th=args.th, q=args.quantile * (0.5**scale), \
                bins=args.bins, s=args.search_range, f=factor)
        
        
        # intensities, variances = estimate_noise_curve(img_0, img_1, w=args.w, T=args.T, th=args.th, q=args.quantile/100, \
                # bins=args.bins, s=args.search_range)


        print()
        print(f"scale {scale} \n")
        print("intensities:")
        print(intensities, "\n")

        print("noise variances:")
        print(variances, "\n")

        print(f"time spent: {time.time() - start} s")

        # save image for result visualization
        # img_0_v = np.transpose(img_0, (1, 2, 0))
        # img_1_v = np.transpose(img_1, (1, 2, 0))
        # cv2.imwrite(f"noisy_0_s{scale}.png", img_0_v.astype(np.uint8))
        # cv2.imwrite(f"noisy_1_s{scale}.png", img_1_v.astype(np.uint8))

        if args.add_noise == True:
            a = args.noise_a # / 4**scale
            b = args.noise_b # / 4**scale

            utils.plot_noise_curve(intensities, variances, a=a, b=b, fname=f"curve_s{scale}.png")

            variances_gt = a + b * intensities

            abs_errors = np.abs(variances_gt - variances)

            print("Absolute errors:")
            print(abs_errors, "\n")

            print("Mean relative error:")
            print((abs_errors / variances_gt).mean(), "\n")

        else:
            utils.plot_noise_curve(intensities, variances, fname=f"curve_s{scale}.png")
        
        scale += 1
    
def main():

    supported_ext = [".tif", ".tiff", ".dng", ".png", "jpg", "jpeg"]

    # verify the two images are in the same extension
    _, extension_0 = os.path.splitext(args.im_0)
    _, extension_1 = os.path.splitext(args.im_1)
    assert extension_0 in supported_ext, \
        f"Only `.tif`, `.tiff`, `.dng`, `.png`, `.jpg` and `.jpeg` formats are support, but `{extension_0}` was found."
    assert extension_0 == extension_1, \
        f"The two input images must be in the same format, but `{extension_0}` and `{extension_1}` were got."
    
    if args.T == -1:
        args.T = args.w + 1

    print("Parameters:")
    print(args)
    print()

    img_0 = utils.read_img(args.im_0, grayscale=args.grayscale)
    img_1 = utils.read_img(args.im_1, grayscale=args.grayscale)

    if args.add_noise == True:
        img_0_noisy = utils.add_noise(img_0, args.noise_a, args.noise_b)
        img_1_noisy = utils.add_noise(img_1, args.noise_a, args.noise_b)

        noise_0 = img_0_noisy - img_0
        noise_1 = img_1_noisy - img_1

        # scale noise for visualization
        max_val = np.max((np.max(noise_0), np.max(noise_1)))
        noise_0 = np.uint8(np.abs(noise_0) / max_val * 255)
        noise_1 = np.uint8(np.abs(noise_1) / max_val * 255)
        print(noise_0.shape)
        noise_0 = np.transpose(noise_0, (1, 2, 0))
        noise_1 = np.transpose(noise_1, (1, 2, 0))
        cv2.imwrite(f"noise_0.png", noise_0)
        cv2.imwrite(f"noise_1.png", noise_1)

        img_0 = img_0_noisy
        img_1 = img_1_noisy

    if img_0.shape != img_1.shape: 
        print("Error: The two input images should have the same size and the same channel")
        quit()

    if args.T > 2 * args.w - 3:
        print("Error: Frequency separator T and block size w should satisfy T<=2*w-3")
        quit()
        

    start = time.time()
    
    # save image for result visualization
    img_0_v = np.transpose(img_0, (1, 2, 0))
    img_1_v = np.transpose(img_1, (1, 2, 0))
    cv2.imwrite(f"noisy_0.png", img_0_v.astype(np.uint8))
    cv2.imwrite(f"noisy_1.png", img_1_v.astype(np.uint8))

    print("###### Output ###### \n")

    if args.multiscale == -1:
        if extension_0 == ".tiff" or extension_0 == ".tif" or extension_0 == ".dng":
            args.multiscale = 1 # no need of multiscale estimation for raw images
        else:
            args.multiscale = 4

    for scale in range(args.multiscale):

        img_0 = img_0.astype(np.float32)
        img_1 = img_1.astype(np.float32)

        # factor = 2**scale

        intensities, variances = estimate_noise_curve_subpixel(img_0, img_1, w=args.w, T=args.T, th=args.th, q=args.quantile * (1**scale), bins=args.bins, s=args.search_range, subpx_order=args.subpx_order, scale=scale)
        

        # intensities, variances = estimate_noise_curve(img_0, img_1, w=args.w, T=args.T, th=args.th, q=args.quantile * (0.25**scale), bins=args.bins, s=args.search_range, scale=scale)

        save_to_txt(intensities, variances, scale)

        print()
        print(f"scale {scale} \n")
        print("intensities:")
        print(intensities, "\n")

        print("noise variances:")
        print(variances, "\n")

        

        # if args.add_noise == True:
        #     a = args.noise_a 
        #     b = args.noise_b
        #     utils.plot_noise_curve(intensities, variances, a=None, b=None, fname=f"curve_s{scale}.png")
        #     variances_gt = a + b * intensities
        #     abs_errors = np.abs(variances_gt - variances)
        #     print("Absolute errors:")
        #     print(abs_errors, "\n")

        #     print("Mean relative error:")
        #     print((abs_errors / variances_gt).mean(), "\n")
        # else:
        
        utils.plot_noise_curve(intensities, variances, fname=f"curve_s{scale}.png")

    print(f"time spent: {time.time() - start} s")

if __name__ == "__main__":
    main()