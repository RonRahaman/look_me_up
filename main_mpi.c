#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

// function to integrate
#define F(x) (x*x)

double rn(unsigned long * seed)
{
  unsigned long n1;
  unsigned long a = 16807;
  unsigned long m = 2147483647;
  n1 = ( a * (*seed) ) % m;
  *seed = n1;
  return (double) n1 / m;
}

int main(int argc, char* argv[]) {

  long n_lookups = 10000000; // number of lookups
  long n_grid = 250000000;   // number of gridpoints
  double * F_vals;           // Discrete values for F(x)
  double interval;           // interval for linearly-spaced grid
  double my_sum = 0;         // Sum of random lookups on F_vals for this proc
  double global_sum = 0;     // Sum of random lookups on F_vals for all procs
  unsigned long seed;        // RNG seed

  double start, end;         // start and end times
  double wall_time;          // wall_time elapsed
  long i, j, k;              // loop control
  double x, f;               // x value and interpolation factor
  int opt;                   // command line option

  int rank;                  // my MPI rank
  int n_procs;               // number of MPI_procs

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get command-line options
  while ((opt = getopt(argc, argv, ":l:g:")) != -1) {
    switch (opt) {
      case 'l': n_lookups = atol(optarg); break;
      case 'g': n_grid    = atol(optarg); break;
      default:  
        fprintf(stderr, "usage: %s [-l n_lookups] [-g n_gridpoints] \n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Print setup
  if (rank == 0)
    printf("Setup:\n  lookups:\t%0.2e\n  gridpoints:\t%0.2e\n  memory:\t%0.2f MB\n  MPI procs:\t%d\n\n", 
        (double) n_lookups, (double) n_grid, (double) n_grid*sizeof(double)/1e6, n_procs);

  // Set interval for grid-spacing
  interval = (double) 1 / (n_grid - 1);

  // Populate values for F(x) on grid
  F_vals = (double *) malloc(n_grid*sizeof(double));
  for (i=0; i<n_grid; i++) {
    F_vals[i] = F(i*interval);
  }

  // Set RNG seed
  seed = rank*19 + 17;

  for (i=rank; i < n_lookups; i += n_procs) {

    // Randomly sample a continous value for x
    x = rn(&seed);

    // Find the indices that bound x on the grid
    j = x / interval;
    k = j+1;

    // Calculate interpolation factor
    f = (k*interval - x) / (k*interval - j*interval);

    my_sum += F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
  }

  MPI_Reduce(&my_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
    printf("Result: %0.6f\n", global_sum / n_lookups);

  free(F_vals);

  MPI_Finalize();

  return 0;
}
