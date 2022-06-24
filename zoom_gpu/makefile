CC=gcc
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart


default: optimized

#convolution.o: ppm.cu ppmKernel.cu
#	$(NVCC) -c -o convolution.o ppm.cu $(NVCC_FLAGS) -D CONV

derbail.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o derbail.o ppm.cu $(NVCC_FLAGS) -D DERBAIL 

optimized.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o optimized.o ppm.cu $(NVCC_FLAGS) -D OPTIMIZED 

#convolution: convolution.o
#	$(NVCC) convolution.o -o convolution $(LD_FLAGS)

derbail: derbail.o
	$(NVCC) derbail.o -o derbail $(LD_FLAGS)

optimized: optimized.o
	$(NVCC) optimized.o -o optimized $(LD_FLAGS)

clean:
	rm -rf *.o optimized derbail outfiles/tmp* *.mp4
