#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "occa_c.h"
#include <sys/time.h>

// function to integrate
#define F(x) (x*x)

const long outer_dim = 128;
const long inner_dim = 128;  // Must be a power of 2 for reduction 

int main(int argc, char* argv[]) {

  //===========================================================================
  //   DECLARATIONS
  //===========================================================================
 
  //---Input parameters--------------------------------------------------------
  // number of lookups
  long n_lookups = (argc < 2) ? 10000000 : atol(argv[1]);  
  // number of gridpoints
  long F_len  = (argc < 3) ? 250000000 : atol(argv[2]);    
  // Discrete values for F(x)
  double *F_vals;
  // interval for linearly-spaced grid
  double interval = (double) 1 / (F_len - 1);
  //---------------------------------------------------------------------------

  //---F_sum-------------------------------------------------------------------
  // These are the sum of the function values, evaluated in kernel.
  // They are the cumulative result of the random lookups.
  
  // Vectors for sums of F(x_i).  Dimensions will be F_sums[0:outer_dim].
  // In kernel, Each outer unit j will reduce is results to F_sum[i].
  // In main, we will need to reduce F_sums to get a single F_sum
  double *F_sums;

  // Sum of all F(x_i) from kernel.  Outside of kernel, F_sums will be reduced
  // to get F_sum
  double F_sum = 0;
  //---------------------------------------------------------------------------

  //---L_sum-------------------------------------------------------------------
  // These are the number of lookups that are actually done on the device
  // They are simple counters, implemented as a sanity check.

  // Will be allocated as L_sums[0:outer_dim].  In kernel, each outer unit j
  // will reduce its results to L_sum[j]
  unsigned long int *L_sums;

  // Cumulative sum of all counts. Outside of kernel, L_sums will be reduced to
  // L_sum
  unsigned long int L_sum = 0;
  //---------------------------------------------------------------------------

  // Loop control
  long i;
  // Timing
  struct timeval start, end;
  double wall_time;

  //---OCCA declarations--------------------------------------------------------
  const char *mode = "CUDA";
  int platformID = 0;
  int deviceID   = 0;

  occaDevice device;
  occaMemory dev_F_sums, dev_F_vals, dev_L_sums;

  occaKernel lookup;

  occaKernelInfo lookupInfo = occaGenKernelInfo();
  occaKernelInfoAddDefine(lookupInfo, "inner_dim", occaLong(inner_dim));
  occaKernelInfoAddDefine(lookupInfo, "outer_dim", occaLong(outer_dim));
  //---------------------------------------------------------------------------

  //===========================================================================
  //   BEFORE KERNEL CALL
  //===========================================================================

  // Print startup message
  printf("Running %0.2e lookups with %0.2e gridpoints in a %0.2f MB array...\n", 
      (double) n_lookups, (double) F_len, (double) F_len*sizeof(double)/1e6);

  // Malloc host memory
  F_vals = (double *) malloc(F_len*sizeof(double));
  F_sums = (double *) calloc( outer_dim, sizeof(double) );
  L_sums = (unsigned long int *) calloc( outer_dim, sizeof(unsigned long int) );

  // Populate values for F(x) on host
  for (i=0; i<F_len; i++) {
    F_vals[i] = F(i*interval);
  }

  // Setup device
  device = occaGetDevice(mode, platformID, deviceID);

  // Setup kernel
  lookup = occaBuildKernelFromSource(device, "lookup.okl", "lookup", lookupInfo);

  // Malloc device memory
  dev_F_vals = occaDeviceMalloc(device, F_len*sizeof(double), NULL);
  dev_F_sums = occaDeviceMalloc(device, outer_dim*sizeof(double), NULL);
  dev_L_sums = occaDeviceMalloc(device, outer_dim*sizeof(unsigned long int), NULL);

  // Host-to-device memcopy
  occaCopyPtrToMem(dev_F_vals, F_vals, F_len*sizeof(double), 0);
  occaCopyPtrToMem(dev_F_sums, F_sums, outer_dim*sizeof(double), 0);
  occaCopyPtrToMem(dev_L_sums, L_sums, outer_dim*sizeof(unsigned long int), 0);

  // Start timing
  occaDeviceFinish(device);
  gettimeofday(&start, NULL);

  //===========================================================================
  //   THE KERNEL CALL
  //===========================================================================

  occaKernelRun( lookup, dev_F_vals, occaDouble(interval), occaLong(n_lookups), dev_F_sums, dev_L_sums);

  //===========================================================================
  //   AFTER KERNEL CALL
  //===========================================================================

  // End timing
  occaDeviceFinish(device);
  gettimeofday(&end, NULL);
  wall_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.;

  // Device-to-host memcopy
  occaCopyMemToPtr(F_sums, dev_F_sums, outer_dim*sizeof(double), 0);
  occaCopyMemToPtr(L_sums, dev_L_sums, outer_dim*sizeof(unsigned long int), 0);

  // Reduce sums
  for (i=0; i<outer_dim; i++) {
    F_sum += F_sums[i];
    L_sum += L_sums[i];
  }

  // Print results
  printf("Lookups in kernel: %lu\n", L_sum);
  printf("Function value: %0.6f\n", F_sum / n_lookups);
  printf("Time:   %0.2e s\n", wall_time);
  printf("Rate:   %0.2e lookups/s\n", n_lookups / wall_time);

  // Cleanup
  occaMemoryFree( dev_F_vals );
  occaMemoryFree( dev_F_sums );
  occaMemoryFree( dev_L_sums );
  occaKernelFree( lookup );
  occaDeviceFree( device );
  free(F_vals);
  free(F_sums);
  free(L_sums);

  return 0;
}
