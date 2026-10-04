"""
Matrix Multiplication Algorithms in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_cpu
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: C[m*N + n] = sum over k of A[m*K + k] * B[k*N + n]
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float result = 0.0f;
            for (int k = 0; k < K; ++k) {
                result += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = result;
        }
    }
}

# Step 2 - max_abs_diff
#include <cmath>

float max_abs_diff(const float* a, const float* b, int n) {
    // TODO: largest |a[i] - b[i]| over n elements, 0 for n == 0
    if (n == 0)
        return 0.0f;
    float max_abs_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_abs_diff) 
            max_abs_diff = diff;
    }
    return max_abs_diff;
}

# Step 3 - matmul_naive_kernel
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: row from threadIdx.x, col from threadIdx.y, guard, accumulate over k, store
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        float dot_product = 0.0f;

        for (int k = 0; k < K; ++k) {
            dot_product += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = dot_product;
    }
}

void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: 16x16 blocks, grid ((M+15)/16, (N+15)/16)
    dim3 block_size(16, 16);
    dim3 grid_size(
        (M + block_size.x - 1) / block_size.x,
        (N + block_size.y - 1) / block_size.y
    );
    matmul_naive_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

# Step 4 - matmul_coalesced_kernel (not yet solved)
# TODO: implement

# Step 5 - time_launch_ms (not yet solved)
# TODO: implement

# Step 6 - matmul_tiled_kernel (not yet solved)
# TODO: implement

# Step 7 - matmul_tiled_1d_kernel (not yet solved)
# TODO: implement

# Step 8 - matmul_tiled_2d_kernel (not yet solved)
# TODO: implement

# Step 9 - matmul_vectorized_kernel (not yet solved)
# TODO: implement

# Step 10 - matmul_double_buffered_kernel (not yet solved)
# TODO: implement

# Step 11 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 12 - matmul_batched_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_splitk_kernel (not yet solved)
# TODO: implement

# Step 14 - gemv_kernel (not yet solved)
# TODO: implement

# Step 15 - matmul_bias_relu_kernel (not yet solved)
# TODO: implement

# Step 16 - matrix_addsub_kernel (not yet solved)
# TODO: implement

# Step 17 - strassen_one_level (not yet solved)
# TODO: implement

# Step 18 - csr_spmm_kernel (not yet solved)
# TODO: implement

# Step 19 - matmul_lower_triangular_kernel (not yet solved)
# TODO: implement

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

