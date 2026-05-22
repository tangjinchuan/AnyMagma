#ifndef REDUCE_H
#define REDUCE_H
// Jinchuan Tang
// 启用双精度浮点扩展（如果设备支持）
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

//==============================================================================
// 单精度求和规约（始终可用）
void magma_ssum_reduce( int n, int i, __local float* x );
void magma_ssum_reduce( int n, int i, __local float* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  barrier(CLK_LOCAL_MEM_FENCE); }
}

//==============================================================================
// 双精度求和规约（仅在设备支持双精度时编译）
#if defined(cl_khr_fp64) || defined(cl_amd_fp64)
void magma_dsum_reduce( int n, int i, __local double* x );
void magma_dsum_reduce( int n, int i, __local double* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  barrier(CLK_LOCAL_MEM_FENCE); }
}
#endif

//==============================================================================
// 单精度复数求和规约（始终可用）
void magma_csum_reduce( int n, int i, __local magmaFloatComplex* x );
void magma_csum_reduce( int n, int i, __local magmaFloatComplex* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+1024] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+ 512] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+ 256] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+ 128] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+  64] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+  32] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+  16] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+   8] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+   4] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+   2] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = MAGMA_C_ADD( x[i], x[i+   1] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
}

//==============================================================================
// 双精度复数求和规约（仅在支持双精度时可用）
#if defined(cl_khr_fp64) || defined(cl_amd_fp64)
void magma_zsum_reduce( int n, int i, __local magmaDoubleComplex* x );
void magma_zsum_reduce( int n, int i, __local magmaDoubleComplex* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+1024] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+ 512] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+ 256] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+ 128] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+  64] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+  32] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+  16] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+   8] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+   4] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+   2] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = MAGMA_Z_ADD( x[i], x[i+   1] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
}
#endif

//==============================================================================
// 单精度最大值规约（始终可用）
void magma_smax_reduce( int n, int i, __local float* x );
void magma_smax_reduce( int n, int i, __local float* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max( x[i], x[i+1024] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max( x[i], x[i+ 512] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max( x[i], x[i+ 256] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max( x[i], x[i+ 128] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max( x[i], x[i+  64] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max( x[i], x[i+  32] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max( x[i], x[i+  16] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max( x[i], x[i+   8] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max( x[i], x[i+   4] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max( x[i], x[i+   2] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max( x[i], x[i+   1] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
}

//==============================================================================
// 双精度最大值规约（仅在支持双精度时可用）
#if defined(cl_khr_fp64) || defined(cl_amd_fp64)
void magma_dmax_reduce( int n, int i, __local double* x );
void magma_dmax_reduce( int n, int i, __local double* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max( x[i], x[i+1024] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max( x[i], x[i+ 512] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max( x[i], x[i+ 256] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max( x[i], x[i+ 128] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max( x[i], x[i+  64] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max( x[i], x[i+  32] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max( x[i], x[i+  16] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max( x[i], x[i+   8] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max( x[i], x[i+   4] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max( x[i], x[i+   2] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max( x[i], x[i+   1] ); }  barrier(CLK_LOCAL_MEM_FENCE); }
}
#endif

#endif // REDUCE_H
