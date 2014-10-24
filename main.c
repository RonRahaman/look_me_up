#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define F(x) (x*x)

int main(int argc, char* argv[]) {

  //number of lookups
  long n_lookups = (argc < 2) ? 10000000 : atol(argv[1]);  
  //number of gridpoints
  long n_grid  = (argc < 3) ? 250000000 : atol(argv[2]);    
  // Discrete values for F(x)
  double * restrict F_vals = (double *) malloc(n_grid*sizeof(double));
  // interval for linearly-spaced grid
  double interval = (double) 1 / (n_grid - 1);

  double sum = 0;

  clock_t start, end;
  double cpu_time;
  double x, f;
  long i, j, k;

  printf("Running %0.2e lookups with %0.2e gridpoints in a %0.2f MB array...\n", 
      (double) n_lookups, (double) n_grid, (double) n_grid*sizeof(double)/1e6);

  // Populate values for F(x) on grid
  for (long i=0; i<n_grid; i++) {
    F_vals[i] = F(i*interval);
  }

  // Seed RNG
  srand((unsigned int) time(NULL));

  start = clock();

  for (i=0; i<n_lookups; i++) {
    // Randomly sample a continous value for x
    x = (double) rand() / RAND_MAX;

    // Find the indices that bound x on the grid
    j = x / interval;
    k = j+1;

    // Calculate interpolation factor
    f = (k*interval - x) / (k*interval - j*interval);

    // Interpolate and accumulate result
    sum += F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
  }

  end = clock();

  cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Result: %0.6f\n", sum / n_lookups);
  printf("Time:   %0.2e s\n", cpu_time);
  printf("Rate:   %0.2e lookups/s\n", n_lookups / cpu_time);

  return 0;
}
