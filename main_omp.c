#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// function to integrate
#define F(x) (x*x)

// threadprivate seed for RNG
unsigned int seed;
#pragma omp threadprivate(seed)

int main(int argc, char* argv[]) {

  long n_lookups = 10000000; // number of lookups
  long n_grid = 250000000;   // number of gridpoints
  int n_threads = 1;         // number of threads
  double * F_vals;           // Discrete values for F(x)
  double interval;           // interval for linearly-spaced grid
  double sum = 0;            // Sum of random lookups on F_vals

  double start, end;         // start and end times
  double wall_time;          // wall_time elapsed
  long i, j, k;              // loop control
  double x, f;               // x value and interpolation factor
  int opt;                   // command line option

  // Get command-line options
  while ((opt = getopt(argc, argv, ":l:t:g:")) != -1) {
    switch (opt) {
      case 'l': n_lookups = atol(optarg); break;
      case 'g': n_grid    = atol(optarg); break;
      case 't': n_threads = atol(optarg); break;
      default:  
        fprintf(stderr, 
            "usage: %s [-l n_lookups] [-g n_gridpoints] [-t n_threads]\n",
            argv[0]);
        return -1;
    }
  }

  // Print setup
  printf("Setup:\n  lookups:\t%0.2e\n  gridpoints:\t%0.2e\n  memory:\t%0.2f MB\n  threads:\t%d\n\n", 
      (double) n_lookups, (double) n_grid, (double) n_grid*sizeof(double)/1e6, n_threads);

  // Set thread numbers
  omp_set_num_threads(n_threads);

  // Set interval for grid-spacing
  interval = (double) 1 / (n_grid - 1);

  // Populate values for F(x) on grid
  F_vals = (double *) malloc(n_grid*sizeof(double));
  for (i=0; i<n_grid; i++) {
    F_vals[i] = F(i*interval);
  }

  // Initialize seeds
#pragma omp parallel
  {
    seed = omp_get_thread_num() * omp_get_wtime() * 1000;
  }

  start = omp_get_wtime();

#pragma omp parallel for \
  default(none) shared(n_lookups,interval,F_vals) private(i,j,k,x,f) \
  schedule(static) \
  reduction(+:sum)
  for (i=0; i<n_lookups; i++) {

    // Randomly sample a continous value for x
    x = (double) rand_r(&seed) / RAND_MAX;

    // Find the indices that bound x on the grid
    j = x / interval;
    k = j+1;

    // Calculate interpolation factor
    f = (k*interval - x) / (k*interval - j*interval);

    // Interpolate and accumulate result
    sum += F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
  }

  end = omp_get_wtime();

  wall_time = end - start;

  printf("Result: %0.6f\n", sum / n_lookups);
  printf("Time:   %0.2e s\n", wall_time);
  printf("Rate:   %0.2e lookups/s\n", n_lookups / wall_time);

  free(F_vals);

  return 0;
}
