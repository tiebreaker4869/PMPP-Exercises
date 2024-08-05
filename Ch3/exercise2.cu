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
void mvKernel(float* v_out, float* m_in, float* v_in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    // out[i] = sum(m[i,j] * v[j])
    float sum = 0;
    for (int i = 0; i < n; i ++) {
      sum += (m_in[idx * n + i] * v_in[i]);
    }

    v_out[idx] = sum;
  }
}

void mv(float* h_out, float* h_m, float* h_v, int n) {
  // allocate memory on device
  float* d_out, *d_m, *d_v;
  cudaMalloc((void**)&d_out, n * sizeof(float)); cudaCheckError();
  cudaMalloc((void**)&d_m, n * n * sizeof(float)); cudaCheckError();
  cudaMalloc((void**)&d_v, n * sizeof(float)); cudaCheckError();

  // copy input to device
  cudaMemcpy(d_m, h_m, n * n * sizeof(float), cudaMemcpyHostToDevice); cudaCheckError();
  cudaMemcpy(d_v, h_v, n * sizeof(float), cudaMemcpyHostToDevice); cudaCheckError();

  // launch kernel
  int blockSize = 32; // 每个块中的线程数
  int numBlocks = (n + blockSize - 1) / blockSize; // 块的数量
  mvKernel<<<numBlocks, blockSize>>>(d_out, d_m, d_v, n); cudaCheckError();

  // copy back to host
  cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost); cudaCheckError();

  // free memory on device
  cudaFree(d_out); cudaCheckError();
  cudaFree(d_m); cudaCheckError();
  cudaFree(d_v); cudaCheckError();
}

int main() {
  int n = 5000;

  float* out = new float[n];
  float* left = new float[n * n];
  float* right = new float[n];

  // Initialize matrices with random values
  for (int i = 0; i < n * n; i++) {
    left[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  for (int i = 0; i < n; i++) {
    right[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  mv(out, left, right, n);

  bool passed = true;

  for (int i = 0; i < n; i ++) {
    float sum = 0;
    
    for (int j = 0; j < n; j ++) {
      sum += (left[i * n + j] * right[j]);
    }
    
    if (fabs(sum - out[i]) > 1e-3) { // Adjusted error tolerance
        std::cout << "Expected: " << sum << ", Actually: " << out[i] << "\n";
        passed = false;
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
