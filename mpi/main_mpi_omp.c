#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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
  int n_threads = 1;         // number of threads
  double * F_vals;           // Discrete values for F(x)
  double interval;           // interval for linearly-spaced grid
  double my_sum = 0;         // Sum of random lookups on F_vals for this proc
  double global_sum = 0;     // Sum of random lookups on F_vals for all procs
  unsigned long seed;        // RNG seed

  double tick, tock;         // start and end times
  double my_wtime;           // wall_time elapsed
  double total_wtime;        // total wall time
  double max_wtime;          // average wall time
  double min_wtime;          // average wall time
  long i, j, k;              // loop control
  double x, f;               // x value and interpolation factor
  int opt;                   // command line option

  int rank;                  // my MPI rank
  int n_procs;               // number of MPI_procs

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get command-line options
  while ((opt = getopt(argc, argv, ":l:g:t:")) != -1) {
    switch (opt) {
      case 'l': n_lookups = atol(optarg); break;
      case 'g': n_grid    = atol(optarg); break;
      case 't': n_threads = atol(optarg); break;
      default:  
                fprintf(stderr, "usage: %s [-l n_lookups] [-g n_gridpoints] [-t n_threads] \n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Set number of threads
  omp_set_num_threads(n_threads);

  // Print setup
  if (rank == 0) {
    printf("Setup:\n");
    printf("  lookups:\t%0.2e\n", (double) n_lookups);
    printf("  gridpoints:\t%0.2e\n", (double) n_grid);
    printf("  memory:\t%0.2f MB\n", (double) n_grid*sizeof(double)/1e6);
    printf("  MPI procs:\t%d\n", n_procs);
    printf("  OMP threads:\t%d\n", n_threads);
  }


  // Set interval for grid-spacing
  interval = (double) 1 / (n_grid - 1);

  // Populate values for F(x) on grid
  F_vals = (double *) malloc(n_grid*sizeof(double));
  for (i=0; i<n_grid; i++) {
    F_vals[i] = F(i*interval);
  }


  tick = MPI_Wtime();

#pragma omp parallel \
  default(none) \
  shared(rank,n_procs, n_lookups,interval,F_vals) \
  private(seed,i,j,k,x,f)\
  reduction(+:my_sum)
  {
    // Set RNG seed
    seed = rank*19 + 17;

#pragma omp for schedule(static)
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
  }

  tock = MPI_Wtime();

  my_wtime = tock - tick;

  MPI_Reduce(&my_sum,   &global_sum,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&my_wtime, &total_wtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&my_wtime, &min_wtime,   1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&my_wtime, &max_wtime,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Results:\n");   
    printf("  Estimator:\t%0.6f\n", global_sum / n_lookups);
    printf("  Min time:\t%0.2e s\n", min_wtime);
    printf("  Max time:\t%0.2e s\n", max_wtime);
    printf("  Avg time:\t%0.2e s\n", total_wtime / n_procs);
    printf("  Avg rate:\t%0.2e lookups/s\n", n_lookups * n_procs / total_wtime);
  }

  free(F_vals);

  MPI_Finalize();

  return 0;
}
