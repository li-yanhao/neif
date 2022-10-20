# Video Signal-Dependent Video Noise Estimation via Inter-frame Prediction


This code is written in python with a part of cython code for acceleration.
The program has been tested in python3.8, and should also work for other versions of python3.


Online demo is available on [IPOL](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000249).

Version 1.0 released on 12/09/2022.

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


## Demo

Estimate noise curve from two successive frames:

frame t             |  frame t+1
:---:|:---:
![](frame0.png)  |  ![](frame1.png)

``` bash
python main.py frame0.png frame1.png
```

The output is:
``` bash
Parameters:
Namespace(T=-1, add_noise=False, bins=16, demosaic=False, im_0='frame0.png', im_1='frame1.png', multiscale=0, noise_a=3, noise_b=3, quantile=5, search_range=5, th=3, w=8)

###### Output ###### 
Parameters:
Namespace(T=-1, add_noise=False, bins=16, demosaic=False, im_0='frame0.png', im_1='frame1.png', multiscale=0, noise_a=3, noise_b=3, quantile=5, search_range=5, th=3, w=8)

###### Output ###### 

intensities:
[[ 10.87425613  14.6869421   17.64328384  20.18724823  22.4411869
   24.70471954  26.92080116  29.29936409  31.69148064  34.67023849
   38.94697571  44.32183456  49.53039932  75.42604065  96.12397766
  140.07905579]] 

noise variances:
[[ 2.75614262  3.46342993  4.06296873  4.23521328  4.80944681  5.42198753
   5.72610235  6.65154648  6.62480831  7.52370214  8.50011253  9.52011967
  10.69717789 18.39974976 20.59931374 29.30751991]] 

time spent: 2.338838815689087 s

```

The plotted noise curve:

<!-- ![](curve.png | width=100) -->


<img src="curve_s0.png" alt="alt text" width="600"/>

# Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{li2022video,
  title={Video Signal-Dependent Video Noise Estimation via Inter-frame Prediction},
  author={Yanhao Li, Marina Gardella, Quentin Bammey, Tina Nikoukhah, Rafael Grompone von Gioi, Miguel Colom, Jean-Michel Morel},
  booktitle={29th IEEE International Conference on Image Processing},
  year={2022}
}
```