#include "helper.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// https://whu-pzhang.github.io/dynamic-allocate-2d-array/

double **alloc_2d_double(size_t n1, size_t n2)
// allocate a 2D ddouble matrix
{
  double **m;
  // allocate pointers to rows (m is actually a pointer to an array)
  if ((m = (double **)malloc(n1 * sizeof(double *))) == NULL) return NULL;
  // allocate rows and set pointers to them
  if ((m[0] = (double *)malloc(n1 * n2 * sizeof(double))) == NULL) return NULL;
  for (size_t i1 = 1; i1 < n1; i1++) m[i1] = m[i1 - 1] + n2;

  return m;
}

float **alloc_2d_float(size_t n1, size_t n2)
// allocate a 2D ddouble matrix
{
  float **m;
  // allocate pointers to rows (m is actually a pointer to an array)
  if ((m = (float **)malloc(n1 * sizeof(float *))) == NULL) return NULL;
  // allocate rows and set pointers to them
  if ((m[0] = (float *)malloc(n1 * n2 * sizeof(float))) == NULL) return NULL;
  for (size_t i1 = 1; i1 < n1; i1++) m[i1] = m[i1 - 1] + n2;

  return m;
}

void free_2d_double(double **m, size_t n1, size_t n2) {
  free(m[0]);
  free(m);
}

void free_2d_float(float **m, size_t n1, size_t n2) {
  free(m[0]);
  free(m);
}

/**
 * Find the best matching positions in the moving frame with subpixel precision.
 *
 * Parameters
 * ----------
 * @ img_ref: Reference image, of size (H, W)
 * @ img_mov: Moving image, of size (H, W)
 * @ pos_ref: Positions of w*w blocks in the reference image, of size (N, 2)
 * @ pos_mov_init: Positions of w*w blocks in the moving image from initial matching, of size (N, 2)
 * @ H: Image height
 * @ W: Image width
 * @ N: Number of block positions
 * @ w: Inner block size
 * @ th: Thickness of surrounding ring
 *
 * Returns
 * -------
 * @ pos_mov_final: Matched block positions in the moving image, of size (N, 2)
 */
void find_best_matching(float *img_ref, float *img_mov, uint16_t *pos_ref, uint16_t *pos_mov_init, int H, int W, int N,
                        int w, int th, int step, uint16_t *pos_mov_final) {
  // printf("H, W, N: %d, %d, %d \n", H, W, N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    float scores[9];

    // N.B.: (r_ref,c_ref) is the top-left coordinate of
    // the outer block containing the surrounding ring
    int r_ref = (int)pos_ref[i * 2] - th;

    int c_ref = (int)pos_ref[i * 2 + 1] - th;

    // Idem. for (r_mov_init, c_mov_init)
    int r_mov_init = (int)pos_mov_init[i * 2] - th;

    int c_mov_init = (int)pos_mov_init[i * 2 + 1] - th;

    // Start patch matching
    int sft_idx = 0;
    for (int r_sft = -1; r_sft <= 1; ++r_sft) {
      for (int c_sft = -1; c_sft <= 1; ++c_sft) {
        const int r_mov = r_mov_init + r_sft;
        const int c_mov = c_mov_init + c_sft;

        float sad = 0;
        for (int ii = 0; ii < th; ii += step) {
          for (int jj = 0; jj < w + 2 * th; jj += step) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        for (int ii = th; ii < th + w; ii += step) {
          for (int jj = 0; jj < th; jj+= step) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        for (int ii = th; ii < th + w; ii += step) {
          for (int jj = w + th; jj < w + 2 * th; jj += step) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        for (int ii = w + th; ii < w + 2 * th; ii += step) {
          for (int jj = 0; jj < w + 2 * th; jj += step) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        scores[sft_idx++] = sad;
      }
    }

    float min_score = scores[0];
    int best_sft_idx = 0;
    for (sft_idx = 0; sft_idx < 9; ++sft_idx) {
      if (min_score > scores[sft_idx]) {
        best_sft_idx = sft_idx;
        min_score = scores[sft_idx];
      }
    }

    int r_sft = best_sft_idx / 3 - 1;
    int c_sft = best_sft_idx % 3 - 1;

    // Output block positions are top-left coordinates of
    // the inner blocks inside their the surrounding rings
    pos_mov_final[i * 2] = r_mov_init + th + r_sft;
    pos_mov_final[i * 2 + 1] = c_mov_init + th + c_sft;
  }
}



void find_best_matching_full(float *img_ref, float *img_mov, uint16_t *pos_ref, uint16_t *pos_mov_init, int H, int W, int N,
                        int w, int th, uint16_t *pos_mov_final) {
  // printf("H, W, N: %d, %d, %d \n", H, W, N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    float scores[9];

    // N.B.: (r_ref,c_ref) is the top-left coordinate of
    // the outer block containing the surrounding ring
    int r_ref = (int)pos_ref[i * 2] - th;

    int c_ref = (int)pos_ref[i * 2 + 1] - th;

    // Idem. for (r_mov_init, c_mov_init)
    int r_mov_init = (int)pos_mov_init[i * 2] - th;

    int c_mov_init = (int)pos_mov_init[i * 2 + 1] - th;

    // Start patch matching
    int sft_idx = 0;
    for (int r_sft = -1; r_sft <= 1; ++r_sft) {
      for (int c_sft = -1; c_sft <= 1; ++c_sft) {
        const int r_mov = r_mov_init + r_sft;
        const int c_mov = c_mov_init + c_sft;

        float sad = 0;
        for (int ii = 0; ii < th; ++ii) {
          for (int jj = 0; jj < w + 2 * th; ++jj) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        for (int ii = th; ii < th + w; ++ii) {
          for (int jj = 0; jj < th; ++jj) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        for (int ii = th; ii < th + w; ++ii) {
          for (int jj = w + th; jj < w + 2 * th; ++jj) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        for (int ii = w + th; ii < w + 2 * th; ++ii) {
          for (int jj = 0; jj < w + 2 * th; ++jj) {
            float dif = img_ref[(r_ref + ii) * W + c_ref + jj] - img_mov[(r_mov + ii) * W + c_mov + jj];
            sad += dif > 0 ? dif : -dif;
          }
        }

        scores[sft_idx++] = sad;
      }
    }

    float min_score = scores[0];
    int best_sft_idx = 0;
    for (sft_idx = 0; sft_idx < 9; ++sft_idx) {
      if (min_score > scores[sft_idx]) {
        best_sft_idx = sft_idx;
        min_score = scores[sft_idx];
      }
    }

    int r_sft = best_sft_idx / 3 - 1;
    int c_sft = best_sft_idx % 3 - 1;

    // Output block positions are top-left coordinates of
    // the inner blocks inside their the surrounding rings
    pos_mov_final[i * 2] = r_mov_init + th + r_sft;
    pos_mov_final[i * 2 + 1] = c_mov_init + th + c_sft;
  }
}