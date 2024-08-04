#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

constexpr int BLUR_SIZE = 7;

// 错误检查宏
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 输入图像是灰度图像
__global__
void blurKernel(unsigned char* Pout, unsigned char* Pin, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    unsigned int pixel_val = 0;
    unsigned int count = 0;
    for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i ++) {
      for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j ++) {
        int cur_row = row + i, cur_col = col + j;
        if (cur_row < 0 || cur_row >= height || cur_col < 0 || cur_col >= height) {
          continue;
        }
        pixel_val += Pin[cur_row * width + cur_col];
        count ++;
      }
    }
    Pout[row * width + col] = pixel_val / count;
  }
}

void blur(const std::string& input_path, const std::string& output_path) {
  cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    return;
  }

  int width = img.cols;
  int height = img.rows;

  // Allocate memory for input and output images on the host
  int size = width * height * sizeof(unsigned char);
  unsigned char* h_in = img.data;
  unsigned char* h_out = new unsigned char[size];

  unsigned char* d_in, *d_out;

  cudaMalloc((void**)&d_in, size); cudaCheckError();
  cudaMalloc((void**)&d_out, size); cudaCheckError();

  // copy input from host to device
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice); cudaCheckError();

  // launch the kernel
  dim3 block_dim(16, 16);
  dim3 grid_dim((width + 15) / 16, (height + 15) / 16);

  blurKernel<<<grid_dim, block_dim>>>(d_out, d_in, width, height); cudaCheckError();

  // copy back to host
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost); cudaCheckError();

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
    std::string input_image_path = "images/output_gray.jpg";
    std::string output_image_path = "images/output_blur.jpg";

    blur(input_image_path, output_image_path);

    std::cout << "Blur complete!" << std::endl;

    return 0;
}