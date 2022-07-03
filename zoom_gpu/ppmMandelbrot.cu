#include "ppm.h"
#include <math.h>

// determines mandelbrot status for each pixel between b1, b2 using optimized (naive) method
__global__ void optimized(coord b1, coord b2, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		double x0 = (tx * (b2.x-b1.x)/x_res) + b1.x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		double y0 = (ty * (b2.y-b1.y)/y_res) + b1.y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)
	
		// determine mandelbrot status of pixel using optimized method
		float x1 = 0;
		float y1 = 0;
		float x2 = 0;
		float y2 = 0;
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

// determines mandelbrot status for each pixel between b1, b2 using derbail method
__global__ void derbail(coord b1, coord b2, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		double x0 = (tx * (b2.x-b1.x)/x_res) + b1.x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		double y0 = (ty * (b2.y-b1.y)/y_res) + b1.y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)
	
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

// determines mandelbrot status for each pixel between b1, b2 using distance estimate method
__global__ void dist_est(coord b1, coord b2, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		double x0 = (tx * (b2.x-b1.x)/x_res) + b1.x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		double y0 = (ty * (b2.y-b1.y)/y_res) + b1.y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)
	
		float x1 = 0;
		float y1 = 0;

		float dx = 1;
		float dy = 0;

		int iteration = 0;
		int max_iteration = 400;

		float limit_radius = 2;
	
		// z'(n+1) = 2 * z'(n) * z(n) + 1
		while (x1*x1 + y1*y1 <= limit_radius*limit_radius && iteration < max_iteration) {
			float xtmp = x1*x1 - y1*y1 + x0; 
			y1 = 2*x1*y1 + y0; 
			x1 = xtmp;
	
			float dxtmp = 2*(dx*x1 - dy*y1) + 1;
			dy = 2*(dy*x1 + dx*y1);
			dx = dxtmp;

			iteration ++; 
		}

		float dist = sqrt(x1*x1 + y1*y1) * 0.5*log((x1*x1 + y1*y1) / (dx*dx + dy*dy));

		// color smooth gradient according to hsv
		outputData[i].red = dist*255;
		outputData[i].green = dist*255;
		outputData[i].blue = dist*255;
	}


}

// determines julia set (of c) status for each pixel between b1, b2
// uses dbail method
__global__ void julia(coord b1, coord b2, coord c, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		double x1 = (tx * (b2.x-b1.x)/x_res) + b1.x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		double y1 = (ty * (b2.y-b1.y)/y_res) + b1.y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)
	
		// determine mandelbrot status of pixel using optimized method
		double x2 = x1*x1;
		double y2 = y1*y1;
		int iteration = 0;
		int max_iteration = 200;
	
		// z(n+1) = z(n)^2 + c
		// for mandelbrot set, instead of z = 0, c = independent
		// for julia set, z = independent, c = constant
		while (x2 + y2 <= 4 && iteration < max_iteration) {
			y1 = 2*x1*y1 + c.y; 
			x1 = x2 - y2 + c.x; 
	
			x2 = x1*x1;
			y2 = y1*y1;
			iteration ++; 
		}
		
		// color smooth gradient according to hsv
		//outputData[i].red = (char)((float)pow((iteration / max_iteration) * 360, 1.5) % 360);
		outputData[i].red = iteration % 255;
		outputData[i].green = 100/ iteration;
		outputData[i].blue = 100 * iteration/ max_iteration;
	}
}
