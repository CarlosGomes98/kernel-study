#pragma once

// Naive GEMM kernel - each thread computes one output element
// C = A * B where A is MxK, B is KxN, C is MxN
__global__ void gemm_naive(float *A, float *B, float *O, int H, int W, int K)
{
    auto xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    auto yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    auto xStride = gridDim.x * blockDim.x;
    auto yStride = gridDim.y * blockDim.y;

    for (int row = yIndex; row < H; row += yStride) {
        for (int col = xIndex; col < W; col += xStride) {
            float tmp = 0.0f;
            for (int inner = 0; inner < K; inner++)
                tmp += A[row * K + inner] * B[inner * W + col];
            O[row * W + col] = tmp;
        }
    }
}

