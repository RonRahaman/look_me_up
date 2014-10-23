#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define F(x) (x*x)

// Binary Search; returns index of closest element that is less than quarry
int binary_search(double * A, double quarry, int n)
{
	int min = 0;
	int max = n-1;
	int mid;
	
	// checks to ensure we're not reading off the end of the grid
	if(A[0] > quarry) return 0;
	else if(A[(n-1)] < quarry) return n-2;
	
	// Begins binary search	
	while(max >= min)
	{
		mid = min + floor((max-min) / 2.0);
		if(A[mid] < quarry) min = mid+1;
		else if(A[mid] > quarry) max = mid-1;
		else return mid;
	}
	return max;
}

int main(int argc, char* argv[]) {

  int lookups = (argc < 2) ? 15000000 : atoi(argv[1]); //number of lookups
  int len  = (argc < 3) ? 11303 : atoi(argv[2]);       //length of grid

  // Discrete values for t
  double * restrict t_vals = (double *) malloc(len*sizeof(double));
  // Discrete values for F(t)
  double * restrict F_vals = (double *) malloc(len*sizeof(double));
  // Storage for results
  double * restrict results = (double *) malloc(lookups*sizeof(double));

  // Create
  double interval = (double) 1 / (len - 1);
  for (long i=0; i<len; i++) {
    t_vals[i] = i*interval;
    F_vals[i] = F(t_vals[i]);
  }

  // Seed RNG
  srand((unsigned int) time(NULL));

  for (int i=0; i<lookups; i++) {
    // Randomly sample a continuous value for t on [0,1]
    double t = (double) rand() / RAND_MAX;
    // Lookup the index of the lower-bounding discrete value
    int j = binary_search(t_vals, t, len);
    // Calculate interpolation factor
    double f = (t_vals[j+1] - t) / (t_vals[j+1] - t_vals[j]);
    // Interpolate and store result
    results[i] = F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
  }

  // Get avg value of results
  double avg = 0;
  for (int i=0; i<lookups; i++) {
    avg = avg + results[i];
  }
  avg = avg/lookups;

  printf("Result: %0.4f\n", avg);

  return 0;
}
