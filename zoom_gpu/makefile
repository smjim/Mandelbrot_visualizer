CC=gcc
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart


default: optimized


optimized.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o optimized.o ppm.cu $(NVCC_FLAGS) -D OPTIMIZED 

derbail.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o derbail.o ppm.cu $(NVCC_FLAGS) -D DERBAIL 

dist.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o dist.o ppm.cu $(NVCC_FLAGS) -D DISTANCE 

julia.o: ppm.cu ppmMandelbrot.cu
	$(NVCC) -c -o julia.o ppm.cu $(NVCC_FLAGS) -D JULIA


optimized: optimized.o
	$(NVCC) optimized.o -o optimized $(LD_FLAGS)

derbail: derbail.o
	$(NVCC) derbail.o -o derbail $(LD_FLAGS)

dist: dist.o
	$(NVCC) dist.o -o dist $(LD_FLAGS)

julia: julia.o
	$(NVCC) julia.o -o julia $(LD_FLAGS)

clean:
	rm -rf *.o optimized derbail julia dist outfiles/tmp* *.mp4
