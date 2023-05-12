# Video Signal-Dependent Noise Estimation via Inter-frame Prediction

This repo is the official implementation of
the ICIP paper
"Video Signal-Dependent Video Noise Estimation via Inter-frame Prediction"
and the IPOL paper "A signal-dependent video noise estimator via inter-frame signal suppression".

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

## Native noise estimator for two raw images

Usage:

``` bash
$ python main.py -h

usage: main.py [-h] [-bins BINS] [-q Q] [-s S] [-w W] [-T T] [-th TH] [-g] [-add_noise] [-noise_a NOISE_A] [-noise_b NOISE_B] im_0 im_1

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
  -g                Whether the input images are in grayscalenoise estimation
  -add_noise        True for adding simulated noise
  -noise_a NOISE_A  Noise model parameter: a
  -noise_b NOISE_B  Noise model parameter: b
```


For instance, estimate noise curves from two successive raw frames:

``` bash
$ python main.py  frame0.tiff  frame1.tiff
```
The noise curves are saved to `curve_s0.txt` with B rows and 2C columns, where the first C columns store the intensities of C channels and the last C columns store the noise variances, and each row is for one bin. The curves are also plotted in `curve_s0.png`:

<img src="readme_img/curve_s0_raw.png" alt="alt text" width="600"/>

## Extended noise estimator for two processed images

Usage:

``` bash
$ python main_v2.py -h

usage: main_v2.py [-h] [-bins BINS] [-q Q] [-s S] [-w W] [-T T] [-th TH] [-g] [-add_noise] [-noise_a NOISE_A] [-noise_b NOISE_B] [-f_us F_US] im_0 im_1

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
  -g                Whether the input images are in grayscale
  -add_noise        True for adding simulated noise
  -noise_a NOISE_A  Noise model parameter: a
  -noise_b NOISE_B  Noise model parameter: b
  -f_us F_US        Upsampling scale for subpixel matching
```


For instance, estimate noise curves from two successive processed images:

``` bash
$ python main.py  frame0.jpg  frame1.jpg
```
The noise estimation will be processed at 4 scales. The noise curves at scale `?` are saved to `curve_s?.txt` with B rows and 2C columns, where the first C columns store the intensities of C channels and the last C columns store the noise variances, and each row is for one bin. 4 scales will be processed The curves are also plotted in `curve_s?.png`:

| scale 0 | scale 1 |   
|:--------------:|:-----------:|
| <img src="readme_img/curve_s0.png" alt="alt text" width="300"/> |  <img src="readme_img/curve_s1.png" alt="alt text" width="300"/> | 
| scale 2 | scale 3 |
| <img src="readme_img/curve_s2.png" alt="alt text" width="300"/> | <img src="readme_img/curve_s3.png" alt="alt text" width="300"/> |



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
