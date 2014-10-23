#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define F(x) (x*x)

int main(int argc, char* argv[]) {

  int n_lookups = (argc < 2) ? 15000000 : atoi(argv[1]);  //number of lookups
  int n_grid  = (argc < 3) ? 11303 : atoi(argv[2]);       //number of gridpoints

  // Discrete values for F(x)
  double * restrict F_vals = (double *) malloc(n_grid*sizeof(double));
  // Storage for results
  double * restrict results = (double *) malloc(n_lookups*sizeof(double));
  // interval for linearly-spaced grid
  double interval = (double) 1 / (n_grid - 1);

  // Populate values for F(x) on grid
  for (long i=0; i<n_grid; i++) {
    F_vals[i] = F(i*interval);
  }

  // Seed RNG
  srand((unsigned int) time(NULL));

  for (int i=0; i<n_lookups; i++) {
    // Randomly sample a continous value for t
    double x = (double) rand() / RAND_MAX;
    // Find lower and upper bounding gridpoints
    int j = x / interval;
    int k = j+1;
    // Calculate interpolation factor
    double f = (k*interval - x) / (k*interval - j*interval);
    // Interpolate and store result
    results[i] = F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
  }

  // Get avg value of results
  double avg = 0;
  for (int i=0; i<n_lookups; i++) {
    avg = avg + results[i];
  }
  avg = avg/n_lookups;

  printf("Result: %0.4f\n", avg);

  return 0;
}
