
import time
from src.estimate import estimate_noise_curve
import src.utils as utils
import cv2
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='Noise Estimation')

parser.add_argument('im_0', type=str,
                    help='First frame filename')
parser.add_argument('im_1', type=str,
                    help='Second frame filename')
parser.add_argument('out', type=str,
                    help='Output curve filename')

parser.add_argument('-bins', type=int, default=16,
                    help='Number of bins')
parser.add_argument('-quantile', type=float, default=5,
                    help='Percentage of quantile, in %')
parser.add_argument('-search_range', type=int, default=5,
                    help='Search range of patch matching')
parser.add_argument('-w', type=int, default=20,
                    help='Block size')
parser.add_argument('-T', type=int, default=21,
                    help='Frequency separator')
parser.add_argument('-th', type=int, default=3,
                    help='Thickness of ring for patch matching')
parser.add_argument('-demosaic', default=False,
                    help='Whether demosaicing is processed before'
                    'noise estimation', action='store_true')
parser.add_argument('-multiscale', type=int, default=0,
                    help='Number of scales for downscaling')
parser.add_argument('-add_noise', default=False,
                    help='True for adding simulated noise',
                    action='store_true')
parser.add_argument('-noise_a', type=float, default=3,
                    help='Noise model parameter: a')
parser.add_argument('-noise_b', type=float, default=3,
                    help='Noise model parameter: b')


args = parser.parse_args()



if __name__ == "__main__":
    print("Parameters:")
    print(args)
    print()
    

    img_0 = utils.read_img(args.im_0, demosaic=args.demosaic)
    img_1 = utils.read_img(args.im_1, demosaic=args.demosaic)

    if args.add_noise == True:
        img_0 = utils.add_noise(img_0, args.noise_a, args.noise_b)
        img_1 = utils.add_noise(img_1, args.noise_a, args.noise_b)

    if img_0.shape != img_1.shape: 
        print("Error: The two input images should have the same size and the same channel")
        quit()

    if args.T > 2 * args.w - 3:
        print("Error: Frequency separator T and block size w should satisfy T<=2*w-3")
        quit()
        

    # if img_0.ndim == 3:
    #     if args.demosaic and img_0.shape[2] > 1:
    #         raise RuntimeError(f'Cannot demosaice a multi-channel image')

    # means, variances = compute_noise_curve(img_0, img_1, bins=args.bins, q=args.quantile / 100, demosaic=args.demosaic,
    #                                        search_range=args.search_range, num_div=num_div, auto_quantile=args.auto_quantile)

    start = time.time()
    
    # args.T = args.w + 1

    scale = 0
    while scale <= args.multiscale:
        if scale > 0:
            # img_0 = utils.downscale(img_0, antialias=False)
            # img_1 = utils.downscale(img_1, antialias=False)

            img_0 = utils.downscale_lebrun(img_0)
            img_1 = utils.downscale_lebrun(img_1)

            # img_0 = utils.downscale_once(img_0)
            # img_1 = utils.downscale_once(img_1)

        img_0 = img_0.astype(np.float32)
        img_1 = img_1.astype(np.float32)

        intensities, variances = estimate_noise_curve(img_0, img_1, w=args.w, T=args.T, th=args.th, q=args.quantile/100 * (0.25**scale), \
                bins=args.bins, s=args.search_range)
        
        variances *= 4**scale
        
        # intensities, variances = estimate_noise_curve(img_0, img_1, w=args.w, T=args.T, th=args.th, q=args.quantile/100, \
                # bins=args.bins, s=args.search_range)


        print("###### Output ###### \n")
        
        print(f"scale {scale}")
        print("intensities:")
        print(intensities, "\n")

        print("noise variances:")
        print(variances, "\n")

        # save image for result visualization
        img_0_v = np.transpose(img_0, (1, 2, 0))
        img_1_v = np.transpose(img_1, (1, 2, 0))
        cv2.imwrite(f"noisy_0_s{scale}.png", img_0_v.astype(np.uint8))
        cv2.imwrite(f"noisy_1_s{scale}.png", img_1_v.astype(np.uint8))

        if args.add_noise == True:
            utils.plot_noise_curve(intensities, variances, a=args.noise_a, b=args.noise_b, fname=args.out)

            variances_gt = args.noise_a + args.noise_b * intensities

            abs_errors = np.abs(variances_gt - variances)

            print("Absolute errors:")
            print(abs_errors, "\n")

            print("Mean relative error:")
            print((abs_errors / variances_gt).mean(), "\n")

        else:
            utils.plot_noise_curve(intensities, variances, fname=f"curve_s{scale}.png")
        
        scale += 1
    
    print(f"time spent: {time.time() - start} s")