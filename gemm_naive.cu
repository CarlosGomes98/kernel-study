#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                                                           \
    {                                                                                                              \
        cudaError_t e = (call);                                                                                    \
        if (e != cudaSuccess)                                                                                      \
        {                                                                                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(1);                                                                                               \
        }                                                                                                          \
    }

#define CHECK_CUBLAS(call)                                                                       \
    {                                                                                            \
        cublasStatus_t s = (call);                                                               \
        if (s != CUBLAS_STATUS_SUCCESS)                                                          \
        {                                                                                        \
            std::cerr << "cuBLAS Error: " << s << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(1);                                                                             \
        }                                                                                        \
    }

__global__ void init(float *A, int size, float val)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        A[i] = val;
}

// matmul kernel. Each thread is responsible for a single output element
__global__ void kernel(float *A, float *B, float *O, int H, int W, int k)
{
    auto xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    auto yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    auto xStride = gridDim.x * blockDim.x;
    auto yStride = gridDim.y * blockDim.y;

    for (int row = yIndex; row < H; row += yStride)
        for (int col = xIndex; col < W; col += xStride)
            for (int inner = 0; inner < k; inner++)
                O[row * W + col] += A[row * k + inner] * B[inner * W + col];
}

int main(void)
{
    cudaFree(0);
    // Increase size so kernel runs noticeably
    const int SIZE = 4096; // try 512, 1024, 2048 depending on your GPU memory
    const size_t bytes = (size_t)SIZE * SIZE * sizeof(float);

    float *A_h = (float *)malloc(bytes), *B_h = (float *)malloc(bytes), *O_h = (float *)malloc(bytes), *O_cublas_h = (float *)malloc(bytes);
    for (int i = 0; i < SIZE * SIZE; ++i)
    {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    float *A_d, *B_d, *O_d;
    CHECK_CUDA(cudaMalloc(&A_d, bytes));
    CHECK_CUDA(cudaMalloc(&B_d, bytes));
    CHECK_CUDA(cudaMalloc(&O_d, bytes));

    // copy host -> device
    CHECK_CUDA(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(O_d, 0, bytes));

    dim3 block(32, 32, 1); // improve this
    dim3 grid((SIZE + block.x - 1) / block.x, (SIZE + block.y - 1) / block.y, 1);

    // Run the custom kernel
    kernel<<<grid, block>>>(A_d, B_d, O_d, SIZE, SIZE, SIZE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // ensure kernel finished
    // Copy the result of the custom kernel back to the host
    cudaMemcpy(O_h, O_d, bytes, cudaMemcpyDeviceToHost);

    // cuBLAS setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS version (on device buffers A_d, B_d -> write into O_d_cu)
    float *O_d_cu;
    CHECK_CUDA(cudaMalloc(&O_d_cu, bytes));
    CHECK_CUDA(cudaMemset(O_d_cu, 0, bytes));

    // Perform matrix multiplication using cuBLAS
    CHECK_CUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                SIZE, SIZE, SIZE,
                &alpha,
                B_d, SIZE, // first operand
                A_d, SIZE, // second operand
                &beta,
                O_d_cu, SIZE));

    // Copy result back to integer matrix
    CHECK_CUDA(cudaMemcpy(O_cublas_h, O_d_cu, bytes, cudaMemcpyDeviceToHost));

    // compare a few elements
    bool match = true;
    for (int r = 0; r < SIZE; ++r)
    {
        for (int c = 0; c < SIZE; ++c)
        {
            float a = O_h[r * SIZE + c];
            float b = O_cublas_h[r * SIZE + c];
            if (fabs(a - b) > 1e-3f)
            {
                match = false;
                break;
            }
        }
        if (!match)
            break;
    }
    std::cout << (match ? "match\n" : "mismatch\n");

    // cleanup
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(O_d);
    cudaFree(O_d_cu);
    free(A_h);
    free(B_h);
    free(O_h);
    free(O_cublas_h);
    return 0;

    return 0;
}