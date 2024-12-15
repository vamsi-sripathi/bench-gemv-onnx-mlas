#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "immintrin.h"
#include "mkl.h"

#define STRINGFY_(x)  #x
#define STRINGFY(x)   STRINGFY_(x)

#define NTRIALS  (100)
#define ALIGN    (64)
#define ERR_TOL  (1.e-02)
#define MAX_INPUT (8)

const uint32_t MlasMaskMoveAvx[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
void MlasSgemmKernelM1Avx(float *p_a, float *p_b, float *p_c, size_t k, size_t n, size_t ldb, float beta);
void IntelSgemmKernelM1Avx2(float *p_a, float *p_b, float *p_c, size_t k, size_t n, size_t ldb, float beta);

int check_results(size_t n, float *p_c_mlas, float *p_c_intel)
{
  for (size_t i=0; i<n; i++) {
    if (fabsf(p_c_mlas[i] - p_c_intel[i]) > ERR_TOL) {
      printf ("[index-%zu]: Expected = %f, Observed = %f\n", i, p_c_mlas[i], p_c_intel[i]);
      fflush(0);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char **argv)
{
  size_t m, n, k, ldb;
  float *p_a, *p_b, *p_c_mlas, *p_c_intel;
  double *t_elasp;
  float beta;

  if (argc != 4) {
    printf ("USAGE: %s n k beta\n", argv[0]); 
    exit(1);
  }

  m = 1;
  n = atoi(argv[1]);
  k = atoi(argv[2]);
  beta = atof(argv[3]);
  ldb = n; //row-major

  p_a = (float *) _mm_malloc(sizeof(float)*m*k, ALIGN);
  p_b = (float *) _mm_malloc(sizeof(float)*k*ldb, ALIGN);
  p_c_mlas  = (float *) _mm_malloc(sizeof(float)*m*n, ALIGN);
  p_c_intel = (float *) _mm_malloc(sizeof(float)*m*n, ALIGN);
  t_elasp = (double *) _mm_malloc(sizeof(double)*NTRIALS, ALIGN);

  for (int i=0; i<k; i++) {
    p_a[i] = rand()/((double) RAND_MAX - 0.5);
  }

  for (int i=0; i<n; i++) {
    p_c_mlas[i] = p_c_intel[i] = rand()/((double) RAND_MAX - 0.5);
  }

  for (int i=0; i<k*n; i++) {
    p_b[i] = rand()/((double) RAND_MAX - 0.5);
  }


  MlasSgemmKernelM1Avx(p_a, p_b, p_c_mlas, k, n, ldb, beta);
  IntelSgemmKernelM1Avx2(p_a, p_b, p_c_intel, k, n, ldb, beta);

  if (check_results (n, p_c_mlas, p_c_intel)) {
    printf ("validation failed!\n");
    exit(1);
  }

  printf ("M = %zu, N = %zu, K = %zu, beta = %.2f: validation passed\n", m, n, k, beta);
  fflush(0);

#ifdef PRINT_MATRICES
  printf ("A-matrix (%zu x %zu):\n", m, k);
  for (int i=0; i<k; i++) {
    printf (" %.2f", p_a[i]); fflush(0);
  }

  printf ("\n\n"); fflush(0);
  printf ("B-matrix (%zu x %zu):\n", k, n);

  for (int i=0; i<k; i++) {
    for (int j=0; j<n; j++) {
      printf (" %.2f", p_b[(i*ldb)+j]); fflush(0);
    }
    printf ("\n"); fflush(0);
  }

  printf ("\n"); fflush(0);
  printf ("C-matrix (%zu x %zu) with Mlas:\n", m, n);
  for (int i=0; i<n; i++) {
    printf (" %.2f", p_c_mlas[i]); fflush(0);
  }

  printf ("\n\n"); fflush(0);
  printf ("C-matrix (%zu x %zu) with Intel:\n", m, n);
  for (int i=0; i<n; i++) {
    printf (" %.2f", p_c_intel[i]); fflush(0);
  }

  printf ("\n"); fflush(0);
#endif

  for (int t=0; t<NTRIALS; t++) {
    t_elasp[t] = dsecnd();
#if FNAME == IntelSgemmKernelM1Avx2
    FNAME(p_a, p_b, p_c_intel, k, n, ldb, beta);
#elif FNAME == MlasSgemmKernelM1Avx
    FNAME(p_a, p_b, p_c_mlas, k, n, ldb, beta);
#endif
    t_elasp[t] = dsecnd() - t_elasp[t];
  }

  double t_best = t_elasp[0];
  double t_avg  = t_elasp[0];
  for (int i=1; i<NTRIALS; i++) {
    if (t_best > t_elasp[i]) {
      t_best = t_elasp[i];
    }
    t_avg += t_elasp[i];
  }
  t_avg /= NTRIALS;

  printf ("%s: M = %zu, N = %zu, K = %zu, t_avg = %.2f, t_best = %.2f\n", STRINGFY(FNAME), m, n, k, 
          t_avg*1.e6, t_best*1.e6);


  _mm_free(p_a);
  _mm_free(p_b);
  _mm_free(p_c_mlas);
  _mm_free(p_c_intel);
  _mm_free(t_elasp);

  return 0;

}
