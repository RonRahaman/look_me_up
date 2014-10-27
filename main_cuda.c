#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

// function to integrate
#define F(x) (x*x)

#pragma acc routine seq
double rn(unsigned long * seed)
{
  double ret;
  unsigned long n1;
  unsigned long a = 16807;
  unsigned long m = 2147483647;
  unsigned long t = a * (*seed);
  n1 = ( a * (*seed) ) % m;
  *seed = n1;
  ret = (double) n1 / m;
  return ret;
}



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

  struct timeval start, end; // start and end times
  double wall_time;  // wall_time elapsed
  long i, j, k;      // loop control
  double x, f;       // x value and interpolation factor
  unsigned long seed; // seed for RNG

  printf("Running %0.2e lookups with %0.2e gridpoints in a %0.2f MB array...\n", 
      (double) n_lookups, (double) n_grid, (double) n_grid*sizeof(double)/1e6);

  // Populate values for F(x) on grid
  for (i=0; i<n_grid; i++) {
    F_vals[i] = F(i*interval);
  }

  gettimeofday(&start, NULL);

  // Initialize seeds
  seed = 1000;

  for (i=0; i<n_lookups; i++) {

    // Randomly sample a continous value for x
    x = (double) rn(&seed);

    // Find the indices that bound x on the grid
    j = x / interval;
    k = j+1;

    // Calculate interpolation factor
    f = (k*interval - x) / (k*interval - j*interval);

    // Interpolate and accumulate result
    sum += F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
  }

  gettimeofday(&end, NULL);

  wall_time = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);

  printf("Result: %0.6f\n", sum / n_lookups);
  printf("Time:   %0.2e s\n", wall_time);
  printf("Rate:   %0.2e lookups/s\n", n_lookups / wall_time);

  return 0;
}
