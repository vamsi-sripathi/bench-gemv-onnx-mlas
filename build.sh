#!/bin/bash

set -x
set -e

rm -f *.o *.bin

icx -O3 -Wall -qmkl -DFNAME="MlasSgemmKernelM1Avx" -c  bench_gemv.c -o bench_gemv_mlas.o
icx -O3 -Wall -xAVX -c MlasSgemmKernelM1Avx.S
icx -O3 -Wall -xAVX2 -c IntelSgemmKernelM1Avx2.c
icx -O3 -Wall -qmkl -z noexecstack bench_gemv_mlas.o  MlasSgemmKernelM1Avx.o IntelSgemmKernelM1Avx2.o  -o gemv_mlas.bin

icx -O3 -Wall -qmkl -DFNAME="IntelSgemmKernelM1Avx2" -c bench_gemv.c -o bench_gemv_intel.o
icx -O3 -Wall -qmkl -z noexecstack bench_gemv_intel.o MlasSgemmKernelM1Avx.o IntelSgemmKernelM1Avx2.o -o gemv_intel.bin

