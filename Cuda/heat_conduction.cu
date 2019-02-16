/*
Requirements: cuda, opencv(compile with ffmpeg)
*/
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#define K 0.25
#define POINT_TO_INDEX(x, y, width) y * width + x;
#define BLOCK_NUM(cnt, nthreads) (cnt + nthreads - 1) / nthreads
using namespace std;
using namespace cv;

__device__ unsigned char temperatureToR(double temperature)
{
    temperature /= 100;
    double red;
    if (temperature <= 66)
        red = 255;
    else
    {
        red = temperature - 60;
        red = 329.698727446 * pow(red, -0.1332047592);
        if (red < 0) red = 0;
        if (red > 255) red = 255;
    }
    return static_cast<unsigned char>(red);
}

__device__ unsigned char temperatureToG(double temperature)
{
    temperature /= 100;
    double green;
    if (temperature <= 66)
    {
        green = temperature;
        green = 99.4708025861 * log(green) - 161.1195681661;
        green = green < 0 ? 0 : green;
        green = green > 255 ? 255 : green;
    }
    else
    {
        green = temperature - 60;
        green = 288.1221695283 * pow(green, -0.0755148492);
        green = green < 0 ? 0 : green;
        green = green > 255 ? 255 : green;
    }
    return static_cast<unsigned char>(green);
}

__device__ unsigned char temperatureToB(int temperature)
{
    temperature /= 100;
    double blue;
    if (temperature >= 68)
        blue = 255;
    else
    {
        if (temperature <= 19)
            blue = 0;
        else
        {
            blue = temperature - 10;
            blue = 138.5177312231 * log(blue) - 305.0447927307;
            blue = blue < 0 ? 0 : blue;
            blue = blue > 255 ? 255 : blue;
        }
    }
    return static_cast<unsigned char>(blue);
}

__global__ void heatChange(double* dest, double* src, int width, int height)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cnt = width * height;
	while(tid < cnt)
	{
		dest[tid] = (1 - 4 * K) * src[tid];
		int x = tid % width;
		int y = tid / width;
		if (x >= 1 & y >= 1) dest[tid] += K * src[tid-width-1];
		if (x <= width - 2 & y >= 1) dest[tid] += K * src[tid-width+1];
		if (x >= 1 & y <= height - 2) dest[tid] += K * src[tid+width-1];
		if (x <= width - 2 & y <= height - 2) dest[tid] += K * src[tid+width+1];
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void heatToColor(unsigned char* color, double* heat, int width, int height)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cnt = width * height;
	while(tid < cnt)
	{
		color[3 * tid] = temperatureToB(heat[tid]);
		color[3 * tid + 1] = temperatureToG(heat[tid]);
		color[3 * tid + 2] = temperatureToR(heat[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

inline double celToKel(double celsius)
{
    return celsius + 273.15;
}

int main(int argc, char *argv[])
{
    const int width = 200;
    const int height = 200;

    unsigned char* heat_array;
    unsigned char* device_heat_array;
    double* temp_array;
    double* device_temp_array_a;
    double* device_temp_array_b;

    // Allocate memory
    // CPU allocation
    heat_array = new unsigned char[width * height * 3];
    temp_array = new double[width * height];
    cout << "CPU memory allocation finished!\n";
    // GPU allocation
    cudaMalloc((void**)&device_heat_array, sizeof(unsigned char) * width * height * 3);
    cudaMalloc((void**)&device_temp_array_a, sizeof(double) * width * height);
    cudaMalloc((void**)&device_temp_array_b, sizeof(double) * width * height);
    cout << "GPU memory allocation finished!\n";
    // ***** modify it later *****
    // Initialize the temperature distribution
    for (int ii = 0; ii < width * height; ii++) temp_array[ii] = 1000;
    for (int ii = 80; ii < 120; ii++)
    {
        for (int jj = 80; jj < 120; jj++)
        {
            temp_array[ii * width + jj] = 40000;
        }
    }
    cout << "Temperature initialization finished!\n";
    // Copy temperature distribution data to GPU memory
    cudaMemcpy(device_temp_array_a, temp_array, sizeof(double) * width * height, cudaMemcpyHostToDevice);
    cout << "Copying CPU data to GPU data finished!\n";
    // Flag for using device_temp_array_a or device_temp_array_b
    // 1 for using b, 0 for using a
    int flag = 1;
    // Some OPENCV stuff to save the video
    Mat heat_mat(Size(width, height), CV_8UC3, static_cast<void *>(heat_array));
    VideoWriter outputVideo;
    outputVideo.open("temp.avi", VideoWriter::fourcc('M','J','P','G') , 60, Size(width, height), true);
    cout << "OPENCV stuff initialization finished!\n";
    for (int ii = 0; ii < 5000; ii++)
    {
    	if (ii % 200 == 0) cout << ii << "th frames finished!\n";
    	if (flag == 0)
    	{
    		heatChange<<<BLOCK_NUM(width * height, 512), 512>>>(device_temp_array_a, device_temp_array_b, width, height);
    		heatToColor<<<BLOCK_NUM(width * height, 512), 512>>>(device_heat_array, device_temp_array_a, width, height);
    		flag = 1;
    	}
    	else
    	{
    		heatChange<<<BLOCK_NUM(width * height, 512), 512>>>(device_temp_array_b, device_temp_array_a, width, height);
    		heatToColor<<<BLOCK_NUM(width * height, 512), 512>>>(device_heat_array, device_temp_array_b, width, height);
    		flag = 0;
    	}
    	// Copy GPU color array to CPU color array
    	cudaMemcpy(heat_array, device_heat_array, sizeof(unsigned char) * width * height * 3, cudaMemcpyDeviceToHost);
        outputVideo << heat_mat;
    }

    // memory deallocation
    // cpu deallocation
    delete[] heat_array;
    delete[] temp_array;
    // gpu_deallocation
    cudaFree(device_heat_array);
    cudaFree(device_temp_array_a);
    cudaFree(device_temp_array_b);

    cout << "finished!\n";
    return 0;
}
