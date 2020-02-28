
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>

#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 

#define CHANNELS 3

__global__ void colorToGreyScaleConversion(unsigned char *pout, unsigned char *pin, int width, int height) {
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	int Row = blockIdx.y*blockDim.y + threadIdx.y;

	if (Col < width && Row < height) {
		int greyoffset = Row * width + Col;
		int rgbOffset = greyoffset * CHANNELS;

		unsigned char r = pin[rgbOffset];
		unsigned char g = pin[rgbOffset + 1];
		unsigned char b = pin[rgbOffset + 2];

		pout[greyoffset] = 0.21f*r + 0.71f*g + 0.07f*b;
	}
}

using namespace cv;
int main(void) {

	// 读入一张图片（缩小图）    
	Mat img = imread("E:\\opencv\\lena512color.tiff");
	// 创建一个名为 "图片"窗口    
	namedWindow("lena");
	// 在窗口中显示图片   
	imshow("lena", img);
	// 等待6000 ms后窗口自动关闭    
	waitKey(6000);

	const int imgheight = img.rows;
	const int imgwidth = img.cols;
	const int imgchannel = img.channels();

	Mat grayImage(imgheight, imgwidth, CV_8UC1, Scalar(0));

	unsigned char *dev_pin;
	unsigned char *dev_pout;

	cudaMalloc((void**)&dev_pin, imgheight*imgwidth*imgchannel* sizeof(unsigned char));
	cudaMalloc((void**)&dev_pout, imgheight*imgwidth*sizeof(unsigned char));

	cudaMemcpy(dev_pin, img.data, imgheight*imgwidth*imgchannel * sizeof(unsigned char), cudaMemcpyHostToDevice);


	dim3 BlockDim(16, 16);
	dim3 GridDim((imgwidth - 1) / BlockDim.x + 1, (imgheight - 1) / BlockDim.y + 1);
	colorToGreyScaleConversion << <GridDim, BlockDim >> > (dev_pout, dev_pin, imgwidth, imgheight);

	cudaMemcpy(grayImage.data, dev_pout, imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(dev_pin);
	cudaFree(dev_pout);
	imshow("grayImage", grayImage);
	waitKey(3000);
	return 0;
}






