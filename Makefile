CC = gcc
OBJECTS = main_serial main_omp main_omp_v2

ifeq ($(CC),gcc)
  CFLAGS = -Wall -O3
  OMP_FLAGS = -fopenmp
endif

ifeq ($(CC),pgcc)
  OBJECTS += main_acc
  CFLAGS = -Minform=inform -fast
  OMP_FLAGS = -mp -Minfo=mp
  ACC_FLAGS = -acc -Minfo=acc 
endif

all: $(OBJECTS)

main_serial: main_serial.c
	$(CC) $(CFLAGS) $^ -o $@

main_omp: main_omp.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $^ -o $@

main_omp_v2: main_omp_v2.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $^ -o $@

main_acc: main_acc.c
	$(CC) $(CFLAGS) $(ACC_FLAGS) $^ -o $@

clean:
	rm -f $(OBJECTS)
