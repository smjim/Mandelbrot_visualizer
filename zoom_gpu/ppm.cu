#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppmMandelbrot.cu"
#include "ppm.h"

const int width = 512;
const int height = 512;

#define RGB_COMPONENT_COLOR 255

#define OUTPUT_TILE_SIZE 12


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

void boundaries(int frame, float *b1x, float *b1y, float *b2x, float *b2y) { 
	b1x = (float*)(-2.00);
	b1y = (float*)(-1.12);
	b2x = (float*)0.47;
	b2y = (float*)1.12;
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

	float b1x;//, float b1y, float b2x, float b2y;
	float b1y;
	float b2x;
	float b2y;
    // for each of the frames run the kernel
    for(i = 0; i < 301; i++) {
        sprintf(outstr, "outfiles/tmp%03d.ppm", i+1);

		boundaries(i, b1x, b1y, b2x, b2y);

#ifdef MANDELBROT
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);
		begin = clock();
		mandelbrot<<<dim_grid, dim_block>>>(b1x, b1y, b2x, b2y, outputData_d, width, height);
		end = clock();
		time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
#endif /*MB*/

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
