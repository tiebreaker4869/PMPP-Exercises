#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// 错误检查宏
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__
void vectorAddKernel(const float* A, const float* B, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

void vectorAdd(const float* A_h, const float* B_h, float* C_h, int N) {
  // allocate memory on device
  float* A_d, *B_d, *C_d;
  int size = N * sizeof(float);
  cudaMalloc((void**)&A_d, size); cudaCheckError();
  cudaMalloc((void**)&B_d, size); cudaCheckError();
  cudaMalloc((void**)&C_d, size); cudaCheckError();

  // copy A and B to device
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice); cudaCheckError();
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice); cudaCheckError();

  // launch kernel 
  int threadPerBlock = 256;
  int blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;
  vectorAddKernel<<<blockPerGrid, threadPerBlock>>>(A_d, B_d, C_d, N);

  // copy result back to host
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost); cudaCheckError();

  // free memory on device
  cudaFree(A_d); cudaCheckError();
  cudaFree(B_d); cudaCheckError();
  cudaFree(C_d); cudaCheckError();
}

int main() {

  int N = 1 << 20;

  float* A = new float[N];
  float* B = new float[N];
  float* C = new float[N];

  for (int i = 0; i < N; i ++) {
    A[i] = static_cast<float>(i);
    B[i] = static_cast<float>(i * 2);
  }

  vectorAdd(A, B, C, N);

  bool passed = true;

  for (int i = 0; i < N; i ++) {
    if (fabs(C[i] - (A[i] + B[i])) > 1e-5) {
      passed = false;
      break;
    }
  }

  if (passed) {
    std::cout << "Passed\n";
  } else {
    std::cout << "Failed\n";
  }


  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}