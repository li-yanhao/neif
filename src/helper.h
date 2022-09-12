// This file is part of the algorithm 
// "A Signal-Dependent Video Noise Estimator via Inter-frame Signal Suppression"


// Copyright (c) 2022 Yanhao Li
// yanhao.li@outlook.com

// This program is free software: you can redistribute it and/or modify it under 
// the terms of the GNU Affero General Public License as published by the Free 
// Software Foundation, either version 3 of the License, or (at your option) any 
// later version.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY 
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License along 
// with this program. If not, see <http://www.gnu.org/licenses/>.


#pragma once

#include <stdint.h>

// #include "numpy/ndarraytypes.h"
// #include "numpy/arrayobject.h"

// #include "numpy/npy_3kcompat.h"
// #include "numpy/ufuncobject.h"

void find_best_matching(float *img_ref, float *img_mov, uint16_t *pos_ref, uint16_t *pos_mov_init, int H, int W, int N,
                        int w, int th, int ups_factor, uint16_t *pos_mov);

void find_best_matching_one_shot(float *img_ref, float *img_ups_mov, uint16_t *pos_ref, uint16_t *pos_mov_init, int H,
                                 int W, int N, int w, int th, int ups_factor, uint16_t *pos_mov_final);