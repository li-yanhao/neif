# Video Signal-Dependent Noise Estimation via Inter-frame Prediction

This repo is the official implementation of
the paper
"Video Signal-Dependent Video Noise Estimation via Inter-frame Prediction". 

## Abstract

We propose a block-based signal-dependent noise estimation method on videos, that leverages inter-frame redundancy to separate noise from signal. Block matching is applied to find block pairs between two consecutive frames with similar signal. Then Ponomarenko’s method is extended by sorting pairs by their low-frequency energy and estimating noise in the high frequencies. Experiments on three datasets show that this method improves on the state of the art.


## About the code

The code is written in python with a part of cython code for acceleration.
The program has been tested both on Linux and MacOS in python3.8, and should also work for other versions of python3.


Online demo is available on [IPOL](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000249).

Version 1.2 released on 12/05/2023.

Contact: Yanhao Li ( yanhao {dot} li {at} outlook {dot} com )


## Install


``` bash
# configure python environment
conda create --name myenv python=3.8
conda activate myenv
pip install -r requirements.txt

# compile cython code
python setup.py build_ext -i
```

## Usage

```
usage: main.py [-h] [-bins BINS] [-q Q] [-s S] [-w W] [-T T] [-th TH] [-g] [-add_noise] [-noise_a NOISE_A] [-noise_b NOISE_B] [-f_us F_US] im_0 im_1

Video Signal-Dependent Noise Estimation via Inter-frame Prediction. (c) 2022 Yanhao Li. Under license GNU AGPL.

positional arguments:
  im_0              First frame filename
  im_1              Second frame filename

optional arguments:
  -h, --help        show this help message and exit
  -bins BINS        Number of bins
  -q Q              Quantile of block pairs
  -s S              Search radius of patch matching
  -w W              Block size
  -T T              Frequency separator
  -th TH            Thickness of ring for patch matching
  -g                Whether the input image is in grayscale
  -add_noise        True for adding simulated noise
  -noise_a NOISE_A  Noise model parameter: a
  -noise_b NOISE_B  Noise model parameter: b
  -f_us F_US        Upsampling scale for subpixel matching
```

Please use the command `python main.py -h` to see the detailed usage.

## Demo

Estimate noise curves from two successive raw frames:

``` bash
python main.py  frame0.tiff  frame1.tiff
```
The noise curves are saved to `curve_s0.txt` with B rows and 2C columns, where the first C columns store the intensities of C channels and the last C columns store the noise variances, and each row is for one bin. The curves are also plotted in `curve_s0.png`:

<img src="readme_img/curve_s0_raw.png" alt="alt text" width="600"/>

Similarly, estimate noise curves from two successive jpg images:

``` bash
python main.py frame0.jpg frame1.jpg 
```

The noise estimation will be processed at 4 scales for .png or .jpg images, the estimated noise curves are like:

| scale 0 | scale 1 |   
|:--------------:|:-----------:|
| <img src="readme_img/curve_s0.png" alt="alt text" width="300"/> |  <img src="readme_img/curve_s1.png" alt="alt text" width="300"/> | 
| scale 2 | scale 3 |
| <img src="readme_img/curve_s2.png" alt="alt text" width="300"/> | <img src="readme_img/curve_s3.png" alt="alt text" width="300"/> |



## Python API

``` python
from src.estimate import estimate_noise_curve

# using default parameters
intensities, variances = estimate_noise_curve(img_ref, img_mov)

# or using custom parameters (see the function description for the use of parameters)
intensities, variances = estimate_noise_curve(img_ref, img_mov, w=..., T=..., th=..., q=..., bins=..., s=..., f=...)

```

Parameters:

- `img_ref`: np.ndarray. Reference image of size (C, H, W)
  
- `img_mov`: np.ndarray. Moving image of size (C, H, W)
    
- `w`: int. Block size
    
- `T`: int. Threshold for separating the entries for low and high frequency DCT coefficents

- `th`: int. Thickness of surrounding ring for matching

- `q`: float. Percentile of blocks used for estimation

- `bins`: int. Number of bins

- `s`: int. Half of search range for patch matching. 
  Note that the range of a squared search region window = search_range * 2 + 1

- `f`: int. The scale factor if estimate noise at a higher scale. Basically the block pairs of size `f*w` are selected, then are downscaled at size `w`, and finally the noise variance is estimated from the high frequencies of the downscaled block pairs.


# Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{li2022video,
  title={Video Signal-Dependent Noise Estimation via Inter-Frame Prediction},
  author={Li, Yanhao and Gardella, Marina and Bammey, Quentin and Nikoukhah, Tina and von Gioi, Rafael Grompone and Colom, Miguel and Morel, Jean-Michel},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1406--1410},
  year={2022},
  organization={IEEE}
}
```
