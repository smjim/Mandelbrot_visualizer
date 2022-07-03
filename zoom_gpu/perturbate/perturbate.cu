#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>  
#include <cmath>
#include <gmpxx.h>
#include "ppmPerturbate.cu"
#include "ppm.h"
#include "precision.h"

using std::vector;

const int width = 1000;
const int height = 1000;

const int length = 101; // length of video in frames

// coords for mandelbrot zooms
//const coord center = {-0.724973/2, -0.357569/2};	// seahorse valley
const coord center = {-0.10109636384562/2, 0.95628651080914/2}; // outer intersection
//const coord center = {-1.6499984109937408174900248316242839345282217233580853461694393097636472584665/2, -0.0000000000000016571246929541869232581096198127918902650429012737576040533449/2};
//const coord center = { 0.360240443437614363236125/2, -0.641313061064803174860375/2};
//const coord center = {-0.72497381200/2, -0.35756855500/2};	// seahorse valley

const double MAX_R = 2.00000000000000000;
const double MIN_R = 0.00000000000000002;

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
#ifdef PERTURBATE 
		boundaries(i, b1, b2);
	    mpf_class zx(center.x, 100); 
	    mpf_class zy(center.y, 100); 

		int depth = 1000;
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
    	printf("%f seconds spent\t---------- frame %d\n", time_spent, i);

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
