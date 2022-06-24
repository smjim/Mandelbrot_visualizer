CC=gcc
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart


default: zoom

#convolution.o: ppm.cu ppmKernel.cu
#	$(NVCC) -c -o convolution.o ppm.cu $(NVCC_FLAGS) -D CONV

zoom.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o zoom.o ppm.cu $(NVCC_FLAGS) -D MANDELBROT 

#convolution: convolution.o
#	$(NVCC) convolution.o -o convolution $(LD_FLAGS)

zoom: zoom.o
	$(NVCC) zoom.o -o zoom $(LD_FLAGS)

clean:
	rm -rf *.o zoom convolution outfiles/tmp* *.mp4
