#pragma once

#include <stdint.h>

// #include "numpy/ndarraytypes.h"
// #include "numpy/arrayobject.h"

// #include "numpy/npy_3kcompat.h"
// #include "numpy/ufuncobject.h"

void find_best_matching(float *img_ref, float *img_mov, uint16_t *pos_ref, uint16_t *pos_mov_init, int H, int W,
                        int N, int w, int th, int ups_factor, uint16_t *pos_mov);

void find_best_matching_one_shot(float *img_ref, float *img_mov, uint16_t *pos_ref, uint16_t *pos_mov_init, int H, int W, int N,
                        int w, int th, int ups_factor, uint16_t *pos_mov_final);