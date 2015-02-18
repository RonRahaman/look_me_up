# Set compilers
CC = gcc
CXX = g++
MPICC = mpicc.openmpi
CUDA_CC = nvcc

# Always compile serial and OMP versions
OBJECTS = main_serial main_omp main_omp_v2 main_mpi

# GCC
ifeq ($(CC),gcc)
  CFLAGS = -Wall -O3
  OMP_FLAGS = -fopenmp
endif

# G++
ifeq ($(CXX),g++)
  CFLAGS = -Wall -O3
  OMP_FLAGS = -fopenmp
  OCCA_FLAGS = -locca
endif

# If using PGCC, also compile ACC version
ifeq ($(CC),pgcc)
  OBJECTS += main_acc
  CFLAGS = -Minform=inform -fast
  OMP_FLAGS = -mp -Minfo=mp
  ACC_FLAGS = -acc -Minfo=acc 
endif

# If using NVCC, also compile CUDA version
ifeq ($(CUDA_CC), nvcc)
  OBJECTS += main_cuda
  CUDA_FLAGS = -arch=sm_35
endif

all: $(OBJECTS)

main_serial: main_serial.c
	$(CC) $(CFLAGS) $^ -o $@

main_mpi: main_mpi.c
	$(MPICC) $(CFLAGS) $^ -o $@

main_omp: main_omp.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $^ -o $@

main_omp_v2: main_omp_v2.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $^ -o $@

main_acc: main_acc.c
	$(CC) $(CFLAGS) $(ACC_FLAGS) $^ -o $@

main_cuda: main_cuda.cu
	$(CUDA_CC) $(CUDA_FLAGS) $^ -o $@

main_occa: main_occa.c 
	$(CC) $(CFLAGS) $^ $(OCCA_FLAGS) -o $@

main_okl: main_okl.c
	$(CC) $(CFLAGS) $^ $(OCCA_FLAGS) -o $@

clean:
	rm -f $(OBJECTS)
