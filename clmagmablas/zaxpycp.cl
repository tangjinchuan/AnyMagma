/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zaxpycp.cu

*/
#include "kernels_header.h"
#include "zaxpycp.h"
/*Enable a specified macro, _z will be replaced by s d z in the generated code for other types.*/
#define PRECISION_z 

#if ( (defined(PRECISION_z) || defined(PRECISION_d)) && (defined(cl_khr_fp64) || defined(cl_amd_fp64)) ) || ( defined(PRECISION_c) || defined(PRECISION_s) )

// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__kernel void
zaxpycp_kernel(
    magma_int_t m,
    __global magmaDoubleComplex *r, unsigned long r_offset,
    __global magmaDoubleComplex *x, unsigned long x_offset,
    __global const magmaDoubleComplex *b, unsigned long b_offset)
{
    r += r_offset;
    x += x_offset;
    b += b_offset;

    const int i = get_local_id(0) + get_group_id(0)*NB;
    if ( i < m ) {
        x[i] = MAGMA_Z_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}
#endif
