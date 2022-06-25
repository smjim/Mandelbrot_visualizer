#include "ppm.h"
#include <math.h>


// determines mandelbrot status for each pixel between b1, b2
__global__ void optimized(double b1x, double b1y, double b2x, double b2y, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		double x0 = (tx * (b2x-b1x)/x_res) + b1x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		double y0 = (ty * (b2y-b1y)/y_res) + b1y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)
	
		// determine mandelbrot status of pixel using optimized method
		double x1 = 0;
		double y1 = 0;
		double x2 = 0;
		double y2 = 0;
		int iteration = 0;
		int max_iteration = 200;
	
		// z(n+1) = z(n)^2 + c
		while (x2 + y2 <= 4 && iteration < max_iteration) {
			y1 = 2*x1*y1 + y0; 
			x1 = x2 - y2 + x0; 
	
			x2 = x1*x1;
			y2 = y1*y1;
			iteration ++; 
		}
		
		// color according to iteration
		outputData[i].red = iteration;
		outputData[i].green = iteration;
		outputData[i].blue = iteration;
	}
}
// determines mandelbrot status for each pixel between b1, b2
__global__ void derbail(double b1x, double b1y, double b2x, double b2y, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		double x0 = (tx * (b2x-b1x)/x_res) + b1x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		double y0 = (ty * (b2y-b1y)/y_res) + b1y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)
	
		float x1 = 0;
		float y1 = 0;
	
		float dx = 1;
		float dy = 0;
		float dx_sum = 0;
		float dy_sum = 0;
	
		int iteration = 0;
		int max_iteration = 400;
	
		int dbail = 1e9; // higher dbail value reveals more detail, increases convergence time
	
		// z'(n+1) = 2 * z'(n) * z(n) + 1
		while (dx_sum*dx_sum + dy_sum*dy_sum < dbail && x1*x1 + y1*y1 <= 4 && iteration < max_iteration) {
			float xtmp = x1*x1 - y1*y1 + x0; 
			y1 = 2*x1*y1 + y0; 
			x1 = xtmp;
	
			float dxtmp = 2*(dx*x1 - dy*y1) + 1;
			dy = 2*(dy*x1 + dx*y1);
			dx = dxtmp;
	
			dx_sum += dx; 
			dy_sum += dy; 
	
			iteration ++; 
		}

		// color smooth gradient according to hsv
		//outputData[i].red = (char)((float)pow((iteration / max_iteration) * 360, 1.5) % 360);
		outputData[i].red = iteration % 360;
		outputData[i].green = 100;
		outputData[i].blue = 100 * iteration/ max_iteration;
	}


}
/*
// determines mandelbrot status for each pixel between b1, b2
__global__ void derbail(double b1x, double b1y, double b2x, double b2y, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		double x0 = (tx * (b2x-b1x)/x_res) + b1x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		double y0 = (ty * (b2y-b1y)/y_res) + b1y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)
	
		float x1 = 0;
		float y1 = 0;
	
		float dx = 1;
		float dy = 0;
		float dx_sum = 0;
		float dy_sum = 0;
	
		int iteration = 0;
		int max_iteration = 400;
	
		int dbail = 1e9; // higher dbail value reveals more detail, increases convergence time
	
		// z'(n+1) = 2 * z'(n) * z(n) + 1
		while (dx_sum*dx_sum + dy_sum*dy_sum < dbail && x1*x1 + y1*y1 <= 4 && iteration < max_iteration) {
			float xtmp = x1*x1 - y1*y1 + x0; 
			y1 = 2*x1*y1 + y0; 
			x1 = xtmp;
	
			float dxtmp = 2*(dx*x1 - dy*y1) + 1;
			dy = 2*(dy*x1 + dx*y1);
			dx = dxtmp;
	
			dx_sum += dx; 
			dy_sum += dy; 
	
			iteration ++; 
		}

		// color according to iteration
		outputData[i].red = iteration;
		outputData[i].green = iteration;
		outputData[i].blue = iteration;
	}


}
*/
