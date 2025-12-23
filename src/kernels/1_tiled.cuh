#pragma once

// Tiled GEMM kernel with shared memory
// Each thread computes one output element
// Uses shared memory tiles to reduce global memory access
template<int BLOCK_SIZE>
__global__ void gemm_tiled(float *A, float *B, float *O, int H, int W, int K)
{
    float tmp = 0.0f;
    __shared__ float tileA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE * BLOCK_SIZE];

    auto xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    auto yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Load tile from A - bounds check
        if (yIndex < H && (threadIdx.x + k) < K) {
            tileA[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[yIndex * K + threadIdx.x + k];
        } else {
            tileA[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
        }
        
        // Load tile from B - bounds check
        if ((threadIdx.y + k) < K && xIndex < W) {
            tileB[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(threadIdx.y + k) * W + xIndex];
        } else {
            tileB[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Compute partial result
        for (int dp = 0; dp < BLOCK_SIZE; dp++) {
            tmp += tileA[threadIdx.y * BLOCK_SIZE + dp] * tileB[dp * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    // Write output with bounds check
    if (yIndex < H && xIndex < W) {
        O[yIndex * W + xIndex] = tmp;
    }
}

