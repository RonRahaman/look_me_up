#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

// function to integrate
#define F(x) (x*x)

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__,__LINE__); exit(-1);} 

const long blocks_per_grid = 128;
const long threads_per_block = 128;  // Must be a power of 2 for reduction 

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

  // Reduction
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
  double *F_vals, *dev_F_vals;
  // interval for linearly-spaced grid
  double interval = (double) 1 / (F_len - 1);
  // Sum of random lookups on F_vals
  double sum = 0;
  long i;

  double *sums, *dev_sums;
  // struct timeval start, end; // start and end times
  // double wall_time;  // wall_time elapsed

  printf("Running %0.2e lookups with %0.2e gridpoints in a %0.2f MB array...\n", 
      (double) n_lookups, (double) F_len, (double) F_len*sizeof(double)/1e6);


  sums = (double *) malloc( blocks_per_grid*sizeof(double) );
  F_vals = (double *) malloc(F_len*sizeof(double));


  // Populate values for F(x) on grid
  for (i=0; i<F_len; i++) {
    F_vals[i] = F(i*interval);
  }

  CUDA_CALL( cudaMalloc( (void**)&dev_sums, blocks_per_grid*sizeof(double) ) );
  CUDA_CALL( cudaMemset( (void*) dev_sums, 0, blocks_per_grid*sizeof(double) ) );


  CUDA_CALL( cudaMalloc( (void**)&dev_F_vals, F_len*sizeof(double)) );
  CUDA_CALL( cudaMemcpy( dev_F_vals, F_vals, F_len*sizeof(double), cudaMemcpyHostToDevice ) );

  lookup<<<blocks_per_grid,threads_per_block>>>(dev_F_vals, F_len, interval, n_lookups, dev_sums);

  CUDA_CALL( cudaMemcpy( sums, dev_sums, threads_per_block*sizeof(double), cudaMemcpyDeviceToHost ));


  // gettimeofday(&start, NULL);


  // gettimeofday(&end, NULL);

  // wall_time = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);

  for (i=0; i<blocks_per_grid; i++) {
    sum += sums[i];
  }

  printf("Result: %0.6f\n", sum / n_lookups);
  // printf("Time:   %0.2e s\n", wall_time);
  // printf("Rate:   %0.2e lookups/s\n", n_lookups / wall_time);

  return 0;
}
