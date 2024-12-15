#include "immintrin.h"

#define K_UNROLL (4)
#define N_UNROLL (32)


// Beta=1. kernel
void IntelSgemmKernelM1Avx2(float *p_a, float *p_b, float *p_c, size_t k, size_t n, size_t ldb, float beta)
{
  size_t k_block = (k/K_UNROLL)*K_UNROLL;
  size_t k_tail  = k - k_block;

  size_t n_block = (n/N_UNROLL)*N_UNROLL;
  size_t n_tail  = n - n_block;

  size_t i, j;

  __m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3,
         ymm_c0, ymm_c1, ymm_c2, ymm_c3;

  __m128 xmm_c0, xmm_c1;

  for (i=0; i<k_block; i+=K_UNROLL) {
    ymm_a0 = _mm256_broadcastss_ps(_mm_load_ss(p_a));
    ymm_a1 = _mm256_broadcastss_ps(_mm_load_ss(p_a + 1));
    ymm_a2 = _mm256_broadcastss_ps(_mm_load_ss(p_a + 2));
    ymm_a3 = _mm256_broadcastss_ps(_mm_load_ss(p_a + 3));

    for (j=0; j<n_block; j+=N_UNROLL) {
      ymm_c0 = _mm256_loadu_ps(p_c);
      ymm_c1 = _mm256_loadu_ps(p_c + 8);
      ymm_c2 = _mm256_loadu_ps(p_c + 16);
      ymm_c3 = _mm256_loadu_ps(p_c + 24);

      // a0 x b00
      ymm_c0 = _mm256_fmadd_ps(ymm_a0, _mm256_loadu_ps(p_b+(i*ldb + j)), ymm_c0);
      // a0 x b01
      ymm_c1 = _mm256_fmadd_ps(ymm_a0, _mm256_loadu_ps(p_b+(i*ldb + j + 8)), ymm_c1);
      // a0 x b02
      ymm_c2 = _mm256_fmadd_ps(ymm_a0, _mm256_loadu_ps(p_b+(i*ldb + j + 16)), ymm_c2);
      // a0 x b03
      ymm_c3 = _mm256_fmadd_ps(ymm_a0, _mm256_loadu_ps(p_b+(i*ldb + j + 24)), ymm_c3);

      // a1 x b10
      ymm_c0 = _mm256_fmadd_ps(ymm_a1, _mm256_loadu_ps(p_b+((i+1)*ldb + j)), ymm_c0);
      // a1 x b11
      ymm_c1 = _mm256_fmadd_ps(ymm_a1, _mm256_loadu_ps(p_b+((i+1)*ldb + j + 8)), ymm_c1);
      // a1 x b12
      ymm_c2 = _mm256_fmadd_ps(ymm_a1, _mm256_loadu_ps(p_b+((i+1)*ldb + j + 16)), ymm_c2);
      // a1 x b13
      ymm_c3 = _mm256_fmadd_ps(ymm_a1, _mm256_loadu_ps(p_b+((i+1)*ldb + j + 24)), ymm_c3);

      // a2 x b20
      ymm_c0 = _mm256_fmadd_ps(ymm_a2, _mm256_loadu_ps(p_b+((i+2)*ldb + j)), ymm_c0);
      // a2 x b21
      ymm_c1 = _mm256_fmadd_ps(ymm_a2, _mm256_loadu_ps(p_b+((i+2)*ldb + j + 8)), ymm_c1);
      // a2 x b22
      ymm_c2 = _mm256_fmadd_ps(ymm_a2, _mm256_loadu_ps(p_b+((i+2)*ldb + j + 16)), ymm_c2);
      // a2 x b23
      ymm_c3 = _mm256_fmadd_ps(ymm_a2, _mm256_loadu_ps(p_b+((i+2)*ldb + j + 24)), ymm_c3);

      // a3 x b30
      ymm_c0 = _mm256_fmadd_ps(ymm_a3, _mm256_loadu_ps(p_b+((i+3)*ldb + j)), ymm_c0);
      // a3 x b31
      ymm_c1 = _mm256_fmadd_ps(ymm_a3, _mm256_loadu_ps(p_b+((i+3)*ldb + j + 8)), ymm_c1);
      // a3 x b32
      ymm_c2 = _mm256_fmadd_ps(ymm_a3, _mm256_loadu_ps(p_b+((i+3)*ldb + j + 16)), ymm_c2);
      // a3 x b33
      ymm_c3 = _mm256_fmadd_ps(ymm_a3, _mm256_loadu_ps(p_b+((i+3)*ldb + j + 24)), ymm_c3);

      _mm256_storeu_ps(p_c,      ymm_c0);
      _mm256_storeu_ps(p_c + 8,  ymm_c1);
      _mm256_storeu_ps(p_c + 16, ymm_c2);
      _mm256_storeu_ps(p_c + 24, ymm_c3);

      /* p_b += N_UNROLL; */
      p_c += N_UNROLL;
    } // n-block

    if (n_tail & 16) {
      ymm_c0 = _mm256_loadu_ps(p_c);
      ymm_c1 = _mm256_loadu_ps(p_c + 8);
      // a0 x b00
      ymm_c0 = _mm256_fmadd_ps(ymm_a0, _mm256_loadu_ps(p_b+(i*ldb + j)), ymm_c0);
      // a0 x b01
      ymm_c1 = _mm256_fmadd_ps(ymm_a0, _mm256_loadu_ps(p_b+(i*ldb + j + 8)), ymm_c1);

      // a1 x b10
      ymm_c0 = _mm256_fmadd_ps(ymm_a1, _mm256_loadu_ps(p_b+((i+1)*ldb + j)), ymm_c0);
      // a1 x b11
      ymm_c1 = _mm256_fmadd_ps(ymm_a1, _mm256_loadu_ps(p_b+((i+1)*ldb + j + 8)), ymm_c1);

      // a2 x b20
      ymm_c0 = _mm256_fmadd_ps(ymm_a2, _mm256_loadu_ps(p_b+((i+2)*ldb + j)), ymm_c0);
      // a2 x b21
      ymm_c1 = _mm256_fmadd_ps(ymm_a2, _mm256_loadu_ps(p_b+((i+2)*ldb + j + 8)), ymm_c1);

      // a3 x b30
      ymm_c0 = _mm256_fmadd_ps(ymm_a3, _mm256_loadu_ps(p_b+((i+3)*ldb + j)), ymm_c0);
      // a3 x b31
      ymm_c1 = _mm256_fmadd_ps(ymm_a3, _mm256_loadu_ps(p_b+((i+3)*ldb + j + 8)), ymm_c1);

      _mm256_storeu_ps(p_c,     ymm_c0);
      _mm256_storeu_ps(p_c + 8, ymm_c1);

      j += 16;
      p_c += 16;

    }

    if (n_tail & 8) {
      ymm_c0 = _mm256_loadu_ps(p_c);
      // a0 x b00
      ymm_c0 = _mm256_fmadd_ps(ymm_a0, _mm256_loadu_ps(p_b+(i*ldb + j)), ymm_c0);

      // a1 x b10
      ymm_c0 = _mm256_fmadd_ps(ymm_a1, _mm256_loadu_ps(p_b+((i+1)*ldb + j)), ymm_c0);

      // a2 x b20
      ymm_c0 = _mm256_fmadd_ps(ymm_a2, _mm256_loadu_ps(p_b+((i+2)*ldb + j)), ymm_c0);

      // a3 x b30
      ymm_c0 = _mm256_fmadd_ps(ymm_a3, _mm256_loadu_ps(p_b+((i+3)*ldb + j)), ymm_c0);

      _mm256_storeu_ps(p_c, ymm_c0);

      j += 8;
      p_c += 8;
    }

    if (n_tail & 4) {
      xmm_c0 = _mm_loadu_ps(p_c);
      // a0 x b00
      xmm_c0 = _mm_fmadd_ps(_mm256_castps256_ps128(ymm_a0),
                            _mm_loadu_ps(p_b+(i*ldb + j)), xmm_c0);

      // a1 x b10
      xmm_c0 = _mm_fmadd_ps(_mm256_castps256_ps128(ymm_a1),
                            _mm_loadu_ps(p_b+((i+1)*ldb + j)), xmm_c0);

      // a2 x b20
      xmm_c0 = _mm_fmadd_ps(_mm256_castps256_ps128(ymm_a2),
                            _mm_loadu_ps(p_b+((i+2)*ldb + j)), xmm_c0);

      // a3 x b30
      xmm_c0 = _mm_fmadd_ps(_mm256_castps256_ps128(ymm_a3),
                            _mm_loadu_ps(p_b+((i+3)*ldb + j)), xmm_c0);

      _mm_storeu_ps(p_c, xmm_c0);

      j += 4;
      p_c += 4;
    }

    if (n_tail & 2 ) {
      xmm_c0 = _mm_load_ss(p_c);
      xmm_c1 = _mm_load_ss(p_c + 1);
      // a0 x b00
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a0),
                            _mm_load_ss(p_b+(i*ldb + j)), xmm_c0);
      // a0 x b01
      xmm_c1 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a0),
                            _mm_load_ss(p_b+(i*ldb + j + 1)), xmm_c1);

      // a1 x b10
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a1),
                            _mm_load_ss(p_b+((i+1)*ldb + j)), xmm_c0);
      // a1 x b11
      xmm_c1 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a1),
                            _mm_load_ss(p_b+((i+1)*ldb + j + 1)), xmm_c1);

      // a2 x b20
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a2),
                            _mm_load_ss(p_b+((i+2)*ldb + j)), xmm_c0);
      // a2 x b21
      xmm_c1 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a2),
                            _mm_load_ss(p_b+((i+2)*ldb + j + 1)), xmm_c1);

      // a3 x b30
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a3),
                            _mm_load_ss(p_b+((i+3)*ldb + j)), xmm_c0);
      // a3 x b31
      xmm_c1 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a3),
                            _mm_load_ss(p_b+((i+3)*ldb + j + 1)), xmm_c1);

      _mm_store_ss(p_c,     xmm_c0);
      _mm_store_ss(p_c + 1, xmm_c1);

      j += 2;
      p_c += 2;
    }

    if (n_tail & 1) {
      xmm_c0 = _mm_load_ss(p_c);
      // a0 x b00
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a0),
                            _mm_load_ss(p_b+(i*ldb + j)), xmm_c0);

      // a1 x b10
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a1),
                            _mm_load_ss(p_b+((i+1)*ldb + j)), xmm_c0);

      // a2 x b20
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a2),
                            _mm_load_ss(p_b+((i+2)*ldb + j)), xmm_c0);

      // a3 x b30
      xmm_c0 = _mm_fmadd_ss(_mm256_castps256_ps128(ymm_a3),
                            _mm_load_ss(p_b+((i+3)*ldb + j)), xmm_c0);

      _mm_store_ss(p_c, xmm_c0);

      j++;
      p_c++;
    }

    p_c -= j;
    p_a += K_UNROLL;
  } // k-block


  // k-tail
  for (j=0; j<n; j++) {
    float tmp = 0.;
    i = k_block;
    for (size_t t=0; t<k_tail; t++) {
      tmp += *(p_a) * p_b[(i*ldb) + j];
      p_a++;
      i++;
    }
    p_a -= k_tail;
    p_c[j] += tmp;
  }
}

#if 0
void IntelSgemmKernelM1_ref(float *p_a, float *p_b, float *p_c, size_t k, size_t n, size_t ldb, float beta)
{
  size_t i, j;
  float a0, a1, a2, a3;
  float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7,
        tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;

  for (i=0; i<k; i+=4) {
    a0 = p_a[i];
    a1 = p_a[i+1];
    a2 = p_a[i+2];
    a3 = p_a[i+3];
    for (j=0; j<n; j+=16) {
      tmp0 = a0 * p_b[(i*ldb) + j];
      tmp1 = a0 * p_b[(i*ldb) + j + 1];
      tmp2 = a0 * p_b[(i*ldb) + j + 2];
      tmp3 = a0 * p_b[(i*ldb) + j + 3];
      tmp4 = a0 * p_b[(i*ldb) + j + 4];
      tmp5 = a0 * p_b[(i*ldb) + j + 5];
      tmp6 = a0 * p_b[(i*ldb) + j + 6];
      tmp7 = a0 * p_b[(i*ldb) + j + 7];

      tmp8  = a0 * p_b[(i*ldb) + j + 8];
      tmp9  = a0 * p_b[(i*ldb) + j + 9];
      tmp10 = a0 * p_b[(i*ldb) + j + 10];
      tmp11 = a0 * p_b[(i*ldb) + j + 11];
      tmp12 = a0 * p_b[(i*ldb) + j + 12];
      tmp13 = a0 * p_b[(i*ldb) + j + 13];
      tmp14 = a0 * p_b[(i*ldb) + j + 14];
      tmp15 = a0 * p_b[(i*ldb) + j + 15];

      tmp0 += a1 * p_b[((i+1)*ldb) + j];
      tmp1 += a1 * p_b[((i+1)*ldb) + j + 1];
      tmp2 += a1 * p_b[((i+1)*ldb) + j + 2];
      tmp3 += a1 * p_b[((i+1)*ldb) + j + 3];
      tmp4 += a1 * p_b[((i+1)*ldb) + j + 4];
      tmp5 += a1 * p_b[((i+1)*ldb) + j + 5];
      tmp6 += a1 * p_b[((i+1)*ldb) + j + 6];
      tmp7 += a1 * p_b[((i+1)*ldb) + j + 7];

      tmp8  += a1 * p_b[((i+1)*ldb) + j + 8];
      tmp9  += a1 * p_b[((i+1)*ldb) + j + 9];
      tmp10 += a1 * p_b[((i+1)*ldb) + j + 10];
      tmp11 += a1 * p_b[((i+1)*ldb) + j + 11];
      tmp12 += a1 * p_b[((i+1)*ldb) + j + 12];
      tmp13 += a1 * p_b[((i+1)*ldb) + j + 13];
      tmp14 += a1 * p_b[((i+1)*ldb) + j + 14];
      tmp15 += a1 * p_b[((i+1)*ldb) + j + 15];

      tmp0 += a2 * p_b[((i+2)*ldb) + j];
      tmp1 += a2 * p_b[((i+2)*ldb) + j + 1];
      tmp2 += a2 * p_b[((i+2)*ldb) + j + 2];
      tmp3 += a2 * p_b[((i+2)*ldb) + j + 3];
      tmp4 += a2 * p_b[((i+2)*ldb) + j + 4];
      tmp5 += a2 * p_b[((i+2)*ldb) + j + 5];
      tmp6 += a2 * p_b[((i+2)*ldb) + j + 6];
      tmp7 += a2 * p_b[((i+2)*ldb) + j + 7];

      tmp8  += a2 * p_b[((i+2)*ldb) + j + 8];
      tmp9  += a2 * p_b[((i+2)*ldb) + j + 9];
      tmp10 += a2 * p_b[((i+2)*ldb) + j + 10];
      tmp11 += a2 * p_b[((i+2)*ldb) + j + 11];
      tmp12 += a2 * p_b[((i+2)*ldb) + j + 12];
      tmp13 += a2 * p_b[((i+2)*ldb) + j + 13];
      tmp14 += a2 * p_b[((i+2)*ldb) + j + 14];
      tmp15 += a2 * p_b[((i+2)*ldb) + j + 15];

      tmp0 += a3 * p_b[((i+3)*ldb) + j];
      tmp1 += a3 * p_b[((i+3)*ldb) + j + 1];
      tmp2 += a3 * p_b[((i+3)*ldb) + j + 2];
      tmp3 += a3 * p_b[((i+3)*ldb) + j + 3];
      tmp4 += a3 * p_b[((i+3)*ldb) + j + 4];
      tmp5 += a3 * p_b[((i+3)*ldb) + j + 5];
      tmp6 += a3 * p_b[((i+3)*ldb) + j + 6];
      tmp7 += a3 * p_b[((i+3)*ldb) + j + 7];

      tmp8  += a3 * p_b[((i+3)*ldb) + j + 8];
      tmp9  += a3 * p_b[((i+3)*ldb) + j + 9];
      tmp10 += a3 * p_b[((i+3)*ldb) + j + 10];
      tmp11 += a3 * p_b[((i+3)*ldb) + j + 11];
      tmp12 += a3 * p_b[((i+3)*ldb) + j + 12];
      tmp13 += a3 * p_b[((i+3)*ldb) + j + 13];
      tmp14 += a3 * p_b[((i+3)*ldb) + j + 14];
      tmp15 += a3 * p_b[((i+3)*ldb) + j + 15];

      p_c[j]     += tmp0;
      p_c[j + 1] += tmp1;
      p_c[j + 2] += tmp2;
      p_c[j + 3] += tmp3;
      p_c[j + 4] += tmp4;
      p_c[j + 5] += tmp5;
      p_c[j + 6] += tmp6;
      p_c[j + 7] += tmp7;
      p_c[j + 8] += tmp8;
      p_c[j + 9] += tmp9;
      p_c[j + 10] += tmp10;
      p_c[j + 11] += tmp11;
      p_c[j + 12] += tmp12;
      p_c[j + 13] += tmp13;
      p_c[j + 14] += tmp14;
      p_c[j + 15] += tmp15;
    }
  }

}
#endif
