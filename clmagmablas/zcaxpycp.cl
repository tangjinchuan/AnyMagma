/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

       auto-converted from zcaxpycp.cu

*/
#include "kernels_header.h"
#include "zcaxpycp.h"

#if ( (defined(PRECISION_z) || defined(PRECISION_d)) && (defined(cl_khr_fp64) || defined(cl_amd_fp64)) ) || ( defined(PRECISION_c) || defined(PRECISION_s) )

// adds   x += r (including conversion to double)  --and--
// copies w = b
// each thread does one index, x[i] and w[i]
__kernel void
zcaxpycp_kernel(
    magma_int_t m,
    __global magmaFloatComplex *r, unsigned long r_offset,
    __global magmaDoubleComplex *x, unsigned long x_offset,
    __global const magmaDoubleComplex *b, unsigned long b_offset,
    __global magmaDoubleComplex *w, unsigned long w_offset )
{
    r += r_offset;
    x += x_offset;
    b += b_offset;
    w += w_offset;

    const int i = get_local_id(0) + get_group_id(0)*NB;
    if ( i < m ) {
        x[i] = MAGMA_Z_ADD( x[i], MAGMA_Z_MAKE( MAGMA_Z_REAL( r[i] ),
                                                MAGMA_Z_IMAG( r[i] ) ) );
        w[i] = b[i];
    }
}
#endif
