#include "ppm.h"
#include <math.h>


// Constant memory for convolution filter
//__constant__ Filter filter_c;

// determines mandelbrot status for each pixel between b1, b2
__global__ void mandelbrot(double b1x, double b1y, double b2x, double b2y, PPMPixel *outputData, int width, int height) {

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
		int max_iteration = 20;
	
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
