CC=gcc
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3 -lgmpxx -lgmp
LD_FLAGS    = -lcudart -lgmpxx -lgmp


default: perturbate 

perturbate.o: perturbate.cu ppmPerturbate.cu precision.h
	$(NVCC) -c -o perturbate.o perturbate.cu $(NVCC_FLAGS) -D PERTURBATE


perturbate: perturbate.o
	$(NVCC) perturbate.o -o perturbate $(LD_FLAGS)

clean:
	rm -rf *.o perturbate outfiles/tmp* *.mp4
