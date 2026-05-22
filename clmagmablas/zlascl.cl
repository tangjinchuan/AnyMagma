/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlascl.cu


       @author Mark Gates
*/
#include "kernels_header.h"
#include "zlascl.h"

#if ( (defined(PRECISION_z) || defined(PRECISION_d)) && (defined(cl_khr_fp64) || defined(cl_amd_fp64)) ) || ( defined(PRECISION_c) || defined(PRECISION_s) )



// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right.
__kernel void
zlascl_full(
    magma_int_t m, magma_int_t n, double mul,
    __global magmaDoubleComplex* A, unsigned long A_offset, magma_int_t lda)
{
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__kernel void
zlascl_lower(
    magma_int_t m, magma_int_t n, double mul,
    __global magmaDoubleComplex* A, unsigned long A_offset, magma_int_t lda)
{
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    int break_d = (ind < n) ? ind : n-1;

    A += ind;
    if (ind < m) {
        for (int j=0; j <= break_d; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__kernel void
zlascl_upper(
    magma_int_t m, magma_int_t n, double mul,
    __global magmaDoubleComplex* A, unsigned long A_offset, magma_int_t lda)
{
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    A += ind;
    if (ind < m) {
        for (int j=n-1; j >= ind; j--)
            A[j*lda] *= mul;
    }
}
#endif
