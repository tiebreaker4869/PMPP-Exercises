#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

// 错误检查宏
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 输入图像的每个像素 rgb 值是一个 0 - 255 的整数，每个像素的 rgb 值连续排列
__global__
void convertKernel(unsigned char* Pout, unsigned char* Pin, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int idx = row * width + col;
    unsigned char r = Pin[idx * 3];
    unsigned char g = Pin[idx * 3 + 1];
    unsigned char b = Pin[idx * 3 + 2];
    // L = 0.21r + 0.71f + 0.07b
    Pout[idx] = 0.21f * r + 0.71f * g + 0.07f * b; 
  }
}

void convertRGB2Grayscale(const std::string& input_path, const std::string& output_path) {
  cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    return;
  }

  int width = img.cols;
  int height = img.rows;

  // Allocate memory for input and output images on the host
  unsigned char* h_in = img.data;
  unsigned char* h_out = new unsigned char[width * height * sizeof(unsigned char)];

  unsigned char* d_in, *d_out;
  int in_size = 3 * width * height * sizeof(unsigned char);
  int out_size = width * height * sizeof(unsigned char);
  cudaMalloc((void**)&d_in, in_size); cudaCheckError();
  cudaMalloc((void**)&d_out, out_size); cudaCheckError();

  // copy input to device
  cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

  // launch kernel
  dim3 block_dim(16, 16);
  dim3 grid_dim((width + 15) / 16, (height + 15) / 16);

  convertKernel<<<grid_dim, block_dim>>>(d_out, d_in, width, height); cudaCheckError();

  // copy result back to host
  cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost); cudaCheckError();

  // write result to file
  cv::Mat gray_img(height, width, CV_8UC1, h_out);

  // Save the grayscale image using OpenCV
  cv::imwrite(output_path, gray_img);
  // deallocate memory on host and device
  cudaFree(d_in); cudaCheckError();
  cudaFree(d_out); cudaCheckError();
  delete[] h_out;
}



int main() {
    std::string input_image_path = "images/input_rgb.jpg";
    std::string output_image_path = "images/output_gray.jpg";

    convertRGB2Grayscale(input_image_path, output_image_path);

    std::cout << "Conversion complete!" << std::endl;

    return 0;
}