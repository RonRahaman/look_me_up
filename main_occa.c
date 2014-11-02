#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "occa_c.h"

// function to integrate
#define F(x) (x*x)

const long outer_dim = 128;
const long inner_dim = 128;  // Must be a power of 2 for reduction 

__device__ double rn(unsigned long * seed)
{
  double ret;
  unsigned long n1;
  unsigned long a = 16807;
  unsigned long m = 2147483647;
  n1 = ( a * (*seed) ) % m;
  *seed = n1;
  ret = (double) n1 / m;
  return ret;
}

__global__ void lookup(double *F_vals, long F_len, double interval, 
    long total_lookups, double *sums) {

  // A per-block cache.  Each thread i writes to sum_cache[i]
  __shared__ double sum_cache[threads_per_block];

  long i,j,k;
  double x, f;
  unsigned long seed;

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int cache_id = threadIdx.x;

  seed = 10000*threadIdx.x + 10* blockIdx.x + threadIdx.x;

  for (i=thread_id; i < total_lookups; i += gridDim.x*blockDim.x) {

    // Randomly sample a continous value for x
    x = (double) rn(&seed);

    // Find the indices that bound x on the grid
    j = x / interval;
    k = j+1;

    // Calculate interpolation factor
    f = (k*interval - x) / (k*interval - j*interval);

    // Interpolate and accumulate result
    sum_cache[cache_id] += F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
  }

  __syncthreads();

  // Naive reduction
  for (i=blockDim.x/2; i != 0; i /= 2) {
    if (cache_id < i)
      sum_cache[cache_id] += sum_cache[cache_id + i];
    __syncthreads();
  }
  if (cache_id == 0) 
    sums[blockIdx.x] = sum_cache[0];

}



int main(int argc, char* argv[]) {

  // number of lookups
  long n_lookups = (argc < 2) ? 10000000 : atol(argv[1]);  
  // number of gridpoints
  long F_len  = (argc < 3) ? 250000000 : atol(argv[2]);    
  // Discrete values for F(x)
  double *F_vals;
  // interval for linearly-spaced grid
  double interval = (double) 1 / (F_len - 1);
  // Sum of random lookups on F_vals
  double sum = 0;
  // Vectors for sums of F(x_i).  Dimensions will be sums[0:blocks_per_grid].
  // Each block j will reduce is results to sum[i].
  double *sums;
  // Timing
  // cudaEvent_t start, stop;
  float elapsed_time;
  // Loop control
  long i;

  const char *mode = "CUDA";
  int platformID = 0;
  int deviceID   = 0;

  occaDevice device;
  occaMemory dev_sums, dev_F_vals;

  int dims = 1;
  occaDim inner_occa_dim, outer_occa_dim;

  occaKernel lookup;

  printf("Running %0.2e lookups with %0.2e gridpoints in a %0.2f MB array...\n", 
      (double) n_lookups, (double) F_len, (double) F_len*sizeof(double)/1e6);

  device = occaGetDevice(mode, platformID, deviceID);
  inner_occa_dim.x = inner_dim;
  outer_occa_dim.x = outer_dim;

  occaKernelSetWorkingDims(lookup, dims, inner_occa_dim, outer_occa_dim);

  dev_sums = occaDeviceMalloc(device, outer_dim*sizeof(double), NULL);
  dev_F_vals = occaDeviceMalloc(device, F_len*sizeof(double), NULL);

  sums = (double *) calloc( outer_dim, sizeof(double) );
  F_vals = (double *) malloc(F_len*sizeof(double));

  // Populate values for F(x) on grid
  for (i=0; i<F_len; i++) {
    F_vals[i] = F(i*interval);
  }

  occaCopyPtrToMem(dev_sums, sums, outer_dim*sizeof(double), 0);
  occaCopyPtrToMem(dev_F_vals, F_vals, F_len*sizeof(double), 0);

  occaKernelRun( lookup, dev_F_vals, occaLong(F_len), occaDouble(interval), occaLong(n_lookups), dev_sums);

  // Copy dev_sums to sums
  occaCopyMemToPtr(sums, dev_sums, outer_dim*sizeof(double), 0);

  // Get cumulative sum
  for (i=0; i<outer_dim; i++) {
    sum += sums[i];
  }

  // Get timings
  printf("Result: %0.6f\n", sum / n_lookups);
  // printf("Time:   %0.2e s\n", elapsed_time);
  // printf("Rate:   %0.2e lookups/s\n", n_lookups / elapsed_time);

  // Cleanup
  // CUDA_CALL( cudaEventDestroy( start ) );
  // CUDA_CALL( cudaEventDestroy( stop ) );
  // CUDA_CALL( cudaFree( dev_F_vals ) );
  // CUDA_CALL( cudaFree( dev_sums ) );
  free(F_vals);
  free(sums);

  return 0;
}
