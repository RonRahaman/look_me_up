CC = gcc
CFLAGS = -Wall -O3

all: main_serial main_omp main_omp_v2

main_serial: main_serial.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

main_omp: main_omp.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ -lm

main_omp_v2: main_omp_v2.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ -lm

clean:
	rm -f main_serial main_omp
