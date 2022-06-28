CC=gcc
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart


default: optimized

julia.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o julia.o ppm.cu $(NVCC_FLAGS) -D JULIA

derbail.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o derbail.o ppm.cu $(NVCC_FLAGS) -D DERBAIL 

optimized.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o optimized.o ppm.cu $(NVCC_FLAGS) -D OPTIMIZED 

julia: julia.o
	$(NVCC) julia.o -o julia $(LD_FLAGS)

derbail: derbail.o
	$(NVCC) derbail.o -o derbail $(LD_FLAGS)

optimized: optimized.o
	$(NVCC) optimized.o -o optimized $(LD_FLAGS)

clean:
	rm -rf *.o optimized derbail julia outfiles/tmp* *.mp4
