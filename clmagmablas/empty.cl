#include "kernels_header.h"

// 单精度版本（始终可用）
__kernel void empty_kernel_float(
    magma_int_t i0, magma_int_t i1, magma_int_t i2, magma_int_t i3, magma_int_t i4,
    magma_int_t i5, magma_int_t i6, magma_int_t i7, magma_int_t i8, magma_int_t i9,
    float d0, float d1, float d2, float d3, float d4,
    __global float *dA,
    __global float *dB,
    __global float *dC)
{
    int tid = get_local_id(0);
    for (int i = 0; i < i0; i++) {
        dC[i + tid] += d1 * dC[i + tid] + d2 * dA[i] * dB[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < i0; i++) {
        dC[i + tid] += d1 * dC[i + tid] + d2 * dA[i] * dB[i];
    }
}

// 双精度版本（仅在支持 double 时编译，但需要扩展启用）
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    __kernel void empty_kernel_double(
        magma_int_t i0, magma_int_t i1, magma_int_t i2, magma_int_t i3, magma_int_t i4,
        magma_int_t i5, magma_int_t i6, magma_int_t i7, magma_int_t i8, magma_int_t i9,
        double d0, double d1, double d2, double d3, double d4,
        __global double *dA,
        __global double *dB,
        __global double *dC)
    {
        int tid = get_local_id(0);
        for (int i = 0; i < i0; i++) {
            dC[i + tid] += d1 * dC[i + tid] + d2 * dA[i] * dB[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < i0; i++) {
            dC[i + tid] += d1 * dC[i + tid] + d2 * dA[i] * dB[i];
        }
    }
#endif
