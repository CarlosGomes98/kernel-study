#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <cublas_v2.h>

#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 8
#define ELEMENTS_PER_THREAD 8
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

// matmul kernel with 1D thread indexing
__global__ void kernel(float *A, float *B, float *O, int H, int W, int K)
{
    // Each thread computes ELEMENTS_PER_THREAD output values
    float tmp[ELEMENTS_PER_THREAD] = {0.0f};
    
    // Shared memory for tiles
    __shared__ float tileA[BLOCK_SIZE_M * BLOCK_SIZE_K];  // 64*8 = 512
    __shared__ float tileB[BLOCK_SIZE_K * BLOCK_SIZE_N];  // 8*64 = 512
    
    // 1D thread index
    int tid = threadIdx.x;  // 0 to 511
    
    // Compute which output elements this thread is responsible for
    // Each thread computes ELEMENTS_PER_THREAD consecutive rows in the same column
    const int threads_per_col = BLOCK_SIZE_M / ELEMENTS_PER_THREAD;
    int local_col = tid % BLOCK_SIZE_N;                              // This thread writes to this column in the output tile
    int local_row_start = (tid / BLOCK_SIZE_N) * ELEMENTS_PER_THREAD; // and to 8 consecutive rows starting from this one
    
    // Global output position
    int global_col = blockIdx.x * BLOCK_SIZE_N + local_col; // This is the above but in the global output matrix
    int global_row_start = blockIdx.y * BLOCK_SIZE_M + local_row_start;
    
    // Loop over K dimension in tiles of BLOCK_SIZE_K
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        
        // Load tileA: 64x8 tile from A
        // Each thread loads one element
        // Map tid to position: row = tid/BK, col = tid%BK
        int tileA_row = tid / BLOCK_SIZE_K;
        int tileA_col = tid % BLOCK_SIZE_K;
        int global_A_row = blockIdx.y * BLOCK_SIZE_M + tileA_row;
        int global_A_col = k + tileA_col;
        
        // Bounds check and load
        if (global_A_row < H && global_A_col < K) {
            tileA[tid] = A[global_A_row * K + global_A_col];
        } else {
            tileA[tid] = 0.0f;
        }
        
        // Load tileB: 8x64 tile from B
        // Each thread loads one element
        // Map tid to position: row = tid/64, col = tid%64
        int tileB_row = tid / BLOCK_SIZE_N;
        int tileB_col = tid % BLOCK_SIZE_N;
        int global_B_row = k + tileB_row;
        int global_B_col = blockIdx.x * BLOCK_SIZE_N + tileB_col;
        
        // Bounds check and load
        if (global_B_row < K && global_B_col < W) {
            tileB[tid] = B[global_B_row * W + global_B_col];
        } else {
            tileB[tid] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute: each thread computes 8 output elements (consecutive rows, same column)
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int local_row = local_row_start + i;
            for (int dp = 0; dp < BLOCK_SIZE_K; dp++) {
                // tileA: access row local_row, column dp
                // tileB: access row dp, column local_col
                // This is easier than it appears. 
                // Remember, this thread is responsible for writing to column local_col, and rows starting from local_row.
                // If we think about it, tileA determines the row, and tileB determines the column.
                // therefore, local_row gives us the starting row for tileA and local_col gives us the starting column for tileB.
                // The only difference is the striding - to access rows in tileA we stride by BLOCK_SIZE_K
                // whereas for rows in the output matrix we stride by BLOCK_SIZE_N
                tmp[i] += tileA[local_row * BLOCK_SIZE_K + dp] * 
                         tileB[dp * BLOCK_SIZE_N + local_col];
            }
        }
        
        __syncthreads();
    }
    
    // Write output (consecutive rows, same column - coalesced!)
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int global_row = global_row_start + i;
        if (global_row < H && global_col < W) {
            O[global_row * W + global_col] = tmp[i];
        }
    }
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

    
    // Each block: 512 threads computing 64x64 output elements
    dim3 block(512, 1, 1);
    // Grid: each block covers BLOCK_SIZE_M rows and BLOCK_SIZE_N columns
    dim3 grid((SIZE + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
              (SIZE + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1);

    // Run the custom kernel
    kernel<<<grid, block>>>(A_d, B_d, O_d, SIZE, SIZE, SIZE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // ensure kernel finished
    // Copy the result of the custom kernel back to the host
    CHECK_CUDA(cudaMemcpy(O_h, O_d, bytes, cudaMemcpyDeviceToHost));

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
