#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>  
#include <cmath>
#include "ppmMandelbrot.cu"
#include "ppm.h"
#include "precision.h"

using std::vector;

const int width = 1000;
const int height = 1000;

const int length = 301; // length of video in frames

// coords for mandelbrot zooms
//const coord center = {-0.6081, -0.6756};
//const coord center = {-0.744749, -0.208039};
//const coord center = {-0.724973, -0.357569};	// seahorse valley
const coord center = {-0.724973/2, -0.357569/2};	// seahorse valley
//const coord center = {-0.10109636384562, 0.95628651080914};
//const coord center = {-0.1010963, 0.9562865};

const double MAX_R = 2.00000000000;
const double MIN_R = 0.00000000002;

// for julia set
//coord center;

#define RGB_COMPONENT_COLOR 255

#define FATAL(...) \
    do {\
        fprintf(stderr, "[%s:%d] \n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //image size
    fprintf(fp, "%d %d\n",width,height);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

// for mandelbrot zooms
void boundaries(int frame, coord &b1, coord &b2) { 
	// logarithmic interpolation between radius MAX_R and MIN_R 
	// first map time domain to log spaced points, then shift
	double t = (double)frame/length;
	double r = pow(MIN_R, t) * pow(MAX_R, 1-t);

	b1.x = center.x - r;
	b1.y = center.y + r;
	b2.x = center.x + r;
	b2.y = center.y - r;
}

// for julia set exploration
void find_center(int frame, coord &center) {
	// define a parametric equation for c
	double t = 6.28*(double)frame/length;

	// formula for radius main cardioid
	center.x = 0.5*cos(t)-0.25*cos(2*t);
	center.y = 0.5*sin(t)-0.25*sin(2*t);
}


int main(){


    clock_t begin, end;
    double time_spent = 0.0;

    char outstr[80];
    int i = 0;


    PPMPixel *outputData_d, *outputData_h;

    cudaError_t cuda_ret;

    // malloc space for output data on host
    outputData_h = (PPMPixel *)malloc(width*height*sizeof(PPMPixel));

    // malloc space for output on device
    cuda_ret = cudaMalloc((void**)&(outputData_d), width*height*sizeof(PPMPixel));
    if(cuda_ret != cudaSuccess) FATAL();	// unable to allocate device memory

    PPMImage *outImage;
    outImage = (PPMImage *)malloc(sizeof(PPMImage));
    outImage->x = width;
    outImage->y = height;

	coord b1 = {center.x + MAX_R, center.y + MAX_R};
	coord b2 = {center.x - MAX_R, center.y - MAX_R};

    // for each of the frames run the kernel
    for(i = 0; i < length; i++) {
        sprintf(outstr, "outfiles/tmp%03d.ppm", i+1);

		begin = clock();
#ifdef DERBAIL
		boundaries(i, b1, b2);
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);

		derbail<<<dim_grid, dim_block>>>(b1, b2, outputData_d, width, height);
#endif /*DB*/

#ifdef OPTIMIZED
		boundaries(i, b1, b2);
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);

		optimized<<<dim_grid, dim_block>>>(b1, b2, outputData_d, width, height);
#endif /*OPT*/

#ifdef DISTANCE
		boundaries(i, b1, b2);
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);

		dist_est<<<dim_grid, dim_block>>>(b1, b2, outputData_d, width, height);
#endif /*DIST*/

#ifdef JULIA
		find_center(i, center);
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);

		julia<<<dim_grid, dim_block>>>(b1, b2, center, outputData_d, width, height);
#endif /*JUL*/

#ifdef PERTURBATE 
		boundaries(i, b1, b2);
	    mpf_class zx(center.x, 100); 
	    mpf_class zy(center.y, 100); 

		int depth = 300;
    	vector<coord> zn(depth);
    	zn = gen_zn(zx, zy, depth);
		int max_iteration = zn.size();

		coord *d_zn;
		size_t bytes = max_iteration*sizeof(coord);
		cudaMalloc(&d_zn, bytes);
		cudaMemcpy(d_zn, zn.data(), bytes, cudaMemcpyHostToDevice);

		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);

		perturbate<<<dim_grid, dim_block>>>(d_zn, max_iteration, b1, b2, outputData_d, width, height);

		cudaFree(d_zn);
#endif /*PT*/

		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    	printf("%f seconds spent ----------- frame %d\n", time_spent, i);

        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL();	// unable to launch/ execute kernel

        cuda_ret = cudaMemcpy(outputData_h, outputData_d, width*height*sizeof(PPMPixel), cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL();	// unable to copy to host


        outImage->data = outputData_h;

        // write processed ppm frame to disk
        writePPM(outstr,outImage);

    }

    // free host and device memory
    free(outputData_h);
    free(outImage);
    cudaFree(outputData_d);
}
