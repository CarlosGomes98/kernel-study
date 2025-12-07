#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

__global__ void init(int size, int n, float *x)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride)
    x[i] = n;
}

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{

  int N = 1 << 22;
  float *x, *y;     // Host pointers
  float *d_x, *d_y; // Device pointers

  // Use cudaHostAlloc instead of malloc for pinned memory
  cudaHostAlloc(&x, N * sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc(&y, N * sizeof(float), cudaHostAllocDefault);

  // Allocate memory on the device
  cudaMalloc(&d_x, N * sizeof(float));
  init<<<8, 32>>>(N, 1, d_x);
  cudaMalloc(&d_y, N * sizeof(float));
  init<<<8, 32>>>(N, 2, d_y);

  add<<<8, 32>>>(N, d_x, d_y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}