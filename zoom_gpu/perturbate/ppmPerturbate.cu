#include "ppm.h"
#include <math.h>

// continuous function in R3 outputs a color (R3) given input t (R1) using parametric function
__device__ PPMPixel colorScheme(const double &t) {
	PPMPixel color;
	// make sure scaled to range (0, 255)
	color.red = 255*sin(t);
	color.green = 255*cos(t); 
	color.blue = 255; 

	return color;
}

// determines mandelbrot set status for pixels between b1, b2 using peturbation series approx.
// theory for iterative process included in doc/sft_maths.pdf
__global__ void perturbate(const coord *zn, int max_iter, const coord b1, const coord b2, 
					PPMPixel *outputData, const int width, const int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;

	int x_res = blockDim.x; // x resolution
	int y_res = gridDim.x;	// y resolution

	if (ty < height && tx < width) {
		// determine complex coords of pixel in relation to b1, b2	
		int i = ty*width + tx;
		coord e;	// point to be operated upon
		e.x = (tx * (b2.x-b1.x)/x_res) + b1.x; // Scaled to lie in Mandelbrot X scale (b1x, b2x)
		e.y = (ty * (b2.y-b1.y)/y_res) + b1.y; // Scaled to lie in Mandelbrot Y scale (b1y, b2y)

	    int iter = 0;
	
	    double en_size;
	
	    // run the iteration loop
	    coord dn = e;   //d0 = e

	    do  
	    {   
	        // dn *= zn[iter] + dn;
	        double tmp = (dn.x*zn[iter].x) - (dn.y*zn[iter].y) + (dn.x*dn.x) - (dn.y*dn.y);
	        dn.y = (dn.x*zn[iter].y) + (dn.y*zn[iter].x) + 2.*(dn.x*dn.y);
	        dn.x = tmp;
	
	        // dn += e;
	        dn = {dn.x + e.x, dn.y + e.y};

	        iter ++; 

	        // en_size = norm(0.5*zn[iter] + dn);
	        en_size = sqrt(pow(0.5*zn[iter].x + dn.x, 2) + pow(0.5*zn[iter].y + dn.y, 2)); 
	    }   
	    while (en_size < 256 && iter < max_iter);

		// coloring algorithm
		double value = pow(iter, 0.5);	// lower exponent == smoother color transitions, less detail
		outputData[i] = colorScheme(value);
		if (iter == max_iter) outputData[i] = {0, 0, 0};
	}
}
