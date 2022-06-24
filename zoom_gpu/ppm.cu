#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppmMandelbrot.cu"
#include "ppm.h"

const int width = 512;
const int height = 512;

const int length = 301; // length of video in frames

const coord startBound_1 = {-2.000, -1.120};
const coord startBound_2 = { 0.470,  1.120};
const coord endBound_1 = {-0.650, 0.600};
const coord endBound_2 = {-0.475, 0.700};

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

/*
void boundaries(int frame, coord *b1, coord *b2) { 
	// linear interpolation between startBound_N and endBound_N
	b1->x = (double)(frame/length) * (endBound_1.x - startBound_1.x) + startBound_1.x;
	b1->y = (double)(frame/length) * (endBound_1.y - startBound_1.y) + startBound_1.y;
	b2->x = (double)(frame/length) * (endBound_2.x - startBound_2.x) + startBound_2.x;
	b2->y = (double)(frame/length) * (endBound_2.y - startBound_2.y) + startBound_2.y;
}
*/
void boundaries(int frame, coord &b1, coord &b2) { 
	// linear interpolation between startBound_N and endBound_N
	b1.x = ((double)frame/length) * (endBound_1.x - startBound_1.x) + startBound_1.x;
	b1.y = ((double)frame/length) * (endBound_1.y - startBound_1.y) + startBound_1.y;
	b2.x = ((double)frame/length) * (endBound_2.x - startBound_2.x) + startBound_2.x;
	b2.y = ((double)frame/length) * (endBound_2.y - startBound_2.y) + startBound_2.y;
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

	coord b1 = startBound_1;
	coord b2 = startBound_2;
    // for each of the frames run the kernel
    for(i = 0; i < length; i++) {
        sprintf(outstr, "outfiles/tmp%03d.ppm", i+1);

		boundaries(i, b1, b2);

#ifdef MANDELBROT
		dim3 dim_grid, dim_block;
		dim_grid = dim3(height, 1,1);
		dim_block = dim3(width, 1,1);
		begin = clock();
		mandelbrot<<<dim_grid, dim_block>>>(b1.x, b1.y, b2.x, b2.y, outputData_d, width, height);
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
