MPICC = mpicc.openmpi
CFLAGS = -Wall -O3
OMP_FLAGS = -fopenmp

OBJECTS = main_mpi main_mpi_omp

all: $(OBJECTS)

main_mpi: main_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@

main_mpi_omp: main_mpi_omp.c
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) $< -o $@

clean:
	rm -f $(OBJECTS)
