#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include "ppmMandelbrot.cu"
#include "ppm.h"

const int width = 1000;
const int height = 1000;

const int length = 301; // length of video in frames

// coords for mandelbrot zooms
//const coord center = {-0.6081, -0.6756};
//const coord center = {-0.744749, -0.208039};
//const coord center = {-0.724973, -0.357569};	// seahorse valley
//const coord center = {-0.10109636384562, 0.95628651080914};
//const coord center = {-0.1010963, 0.9562865};
const coord center = {0.0, 0.0};

const double MAX_R = 2.00000000000;
const double MIN_R = 0.00000000002;

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
void find_center(int frame, coord &c) {
	// define a parametric equation for c
	double t = 6.28*(double)frame/length;

	// formula for main cardioid
	c.x = 0.5*cos(t)-0.25*cos(2*t);
	c.y = 0.5*sin(t)-0.25*sin(2*t);
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
	coord c;

    // for each of the frames run the kernel
    for(i = 0; i < length; i++) {
        sprintf(outstr, "outfiles/tmp%03d.ppm", i+1);

#ifdef DERBAIL
		boundaries(i, b1, b2);
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);
		begin = clock();
		derbail<<<dim_grid, dim_block>>>(b1, b2, outputData_d, width, height);
		end = clock();
		time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
#endif /*DB*/

#ifdef OPTIMIZED
		boundaries(i, b1, b2);
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);
		begin = clock();
		optimized<<<dim_grid, dim_block>>>(b1, b2, outputData_d, width, height);
		end = clock();
		time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
#endif /*OPT*/

#ifdef JULIA
		find_center(i, c);
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);
		begin = clock();
		julia<<<dim_grid, dim_block>>>(b1, b2, c, outputData_d, width, height);
		end = clock();
		time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
#endif /*JUL*/

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

    printf("%f seconds spent\n", time_spent);

}
