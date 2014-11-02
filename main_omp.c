#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// function to integrate
#define F(x) (x*x)

// threadprivate seed for RNG
unsigned int seed;
#pragma omp threadprivate(seed)

int main(int argc, char* argv[]) {

  // number of lookups
  long n_lookups = (argc < 2) ? 10000000 : atol(argv[1]);  
  // number of gridpoints
  long n_grid  = (argc < 3) ? 250000000 : atol(argv[2]);    
  // Discrete values for F(x)
  double * F_vals = (double *) malloc(n_grid*sizeof(double));
  // interval for linearly-spaced grid
  double interval = (double) 1 / (n_grid - 1);
  // Sum of random lookups on F_vals
  double sum = 0;

  double start, end; // start and end times
  double wall_time;  // wall_time elapsed
  long i, j, k;      // loop control
  double x, f;       // x value and interpolation factor

  printf("Running %0.2e lookups with %0.2e gridpoints in a %0.2f MB array...\n", 
      (double) n_lookups, (double) n_grid, (double) n_grid*sizeof(double)/1e6);

  // Populate values for F(x) on grid
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
