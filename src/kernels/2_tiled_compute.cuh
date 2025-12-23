#pragma once

// Tiled GEMM kernel with increased compute per thread
// Each thread computes ELEMENTS_PER_THREAD consecutive output elements
// Uses 1D thread indexing and shared memory
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int ELEMENTS_PER_THREAD>
__global__ void gemm_tiled_compute(float *A, float *B, float *O, int H, int W, int K)
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

