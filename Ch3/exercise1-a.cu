#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// 错误检查宏
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__
void matrixMulKernel(float* m_out, float* m_left, float* m_right, int m, int n, int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < m) {
    for (int col = 0; col < k; col++) {
      float sum = 0;
      for (int t = 0; t < n; t++) {
        sum += (m_left[row * n + t] * m_right[t * k + col]);
      }
      m_out[row * k + col] = sum;
    }
  }
}

void matrixMul(float* h_out, float* h_left, float* h_right, int m, int n, int k) {
  // allocate memory on device
  float* d_out, *d_left, *d_right;
  cudaMalloc((void**)&d_out, m * k * sizeof(float)); cudaCheckError();
  cudaMalloc((void**)&d_left, m * n * sizeof(float)); cudaCheckError();
  cudaMalloc((void**)&d_right, n * k * sizeof(float)); cudaCheckError();

  // copy input to device
  cudaMemcpy(d_left, h_left, m * n * sizeof(float), cudaMemcpyHostToDevice); cudaCheckError();
  cudaMemcpy(d_right, h_right, n * k * sizeof(float), cudaMemcpyHostToDevice); cudaCheckError();

  // launch kernel
  int blockSize = 32; // 每个块中的线程数
  int numBlocks = (m + blockSize - 1) / blockSize; // 块的数量
  matrixMulKernel<<<numBlocks, blockSize>>>(d_out, d_left, d_right, m, n, k); cudaCheckError();

  // copy back to host
  cudaMemcpy(h_out, d_out, m * k * sizeof(float), cudaMemcpyDeviceToHost); cudaCheckError();

  // free memory on device
  cudaFree(d_out); cudaCheckError();
  cudaFree(d_left); cudaCheckError();
  cudaFree(d_right); cudaCheckError();
}

int main() {
  int m = 1000, n = 2000, k = 1500;

  float* out = new float[m * k];
  float* left = new float[m * n];
  float* right = new float[n * k];

  // Initialize matrices with random values
  for (int i = 0; i < m * n; i++) {
    left[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  for (int i = 0; i < n * k; i++) {
    right[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  matrixMul(out, left, right, m, n, k);

  bool passed = true;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      float sum = 0;
      for (int t = 0; t < n; t++) {
        sum += (left[i * n + t] * right[t * k + j]);
      }
      if (fabs(sum - out[i * k + j]) > 1e-3) { // Adjusted error tolerance
        std::cout << "Expected: " << sum << ", Actually: " << out[i * k + j] << "\n";
        passed = false;
        break;
      }
    }
    if (!passed) {
      break;
    }
  }

  if (passed) {
    std::cout << "Passed\n";
  } else {
    std::cout << "Failed\n";
  }

  delete[] out;
  delete[] left;
  delete[] right;

  return 0;
}
