#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <openacc.h>

// function to integrate
#define F(x) (x*x)
#define RN_S_SEED 1337

#pragma acc routine seq
extern int rand();


// Compiler does not support mod operator for unsigned longs
// double rn(unsigned long * seed)
#pragma acc routine seq
double rn(unsigned seed)
{
  double ret;
  // unsigned long n1;
  // unsigned long a = 16807;
  // unsigned long m = 2147483647;
  // unsigned long t = a * (*seed);
  unsigned n1;
  unsigned a = 16807;
  unsigned m = 2147483647;
  unsigned t = a * seed;
  n1 = ( a * seed ) % m;
  ret = (double) n1 / m;
  return ret;
}
// double rn(unsigned * seed)
// {
//   double ret;
//   // unsigned long n1;
//   // unsigned long a = 16807;
//   // unsigned long m = 2147483647;
//   // unsigned long t = a * (*seed);
//   unsigned n1;
//   unsigned a = 16807;
//   unsigned m = 2147483647;
//   unsigned t = a * (*seed);
//   n1 = ( a * (*seed) ) % m;
//   *seed = n1;
//   ret = (double) n1 / m;
//   return ret;
// }

// A stateless random number generator, 
//
// Produces the n-th element of a sequence that was initially seeded with 'seed'.  
//
// This RNG yields a reproducible stream of numbers without a static seed.  On
// parallel architectures, it may be more convenient than rn_v(). Calls
// to rn_v() must be protected by critical sections, but calls to rn_s() do not
// require this protection. 
//
// Based on algorithms from Press et al., "Numerical Recipes", second ed.  
// The bitmasks assume unsigned long ints are 32 bits; and that 32-bit floats
// follow IEEE representation.  See "Numerical Recipes" for details.  
//#pragma acc routine seq
float rn_s(const long seed, const long n) 
{
  // For pdes
  unsigned long i,ia,ib,iswap,itmph=0,itmpl=0; 
  const unsigned long c1[4]={ 0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L}; 
  const unsigned long c2[4]={ 0x4b0f3b58L, 0xe874f0c3L, 0x6955c5a6L, 0x55a7ca46L};
  // For ran4
  unsigned long irword,itemp,lword;
  const unsigned long jflone = 0x3f800000; 
  const unsigned long jflmsk = 0x007fffff;
  irword = n;
  lword = seed;
  // Run pdes
  for (i=0; i<4; i++) {
    ia = (iswap = irword) ^ c1[i];
    itmpl = ia & 0xffff;
    itmph = ia >> 16;
    ib = itmpl*itmpl+ ~(itmph*itmph); 
    irword = lword ^ (((ia = (ib >> 16) |
            ((ib & 0xffff) << 16)) ^ c2[i])+itmpl*itmph); 
    lword = iswap;
  }
  itemp=jflone | (jflmsk & irword); 
  return (*(float *)&itemp)-1.0;
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
  unsigned seed; // seed for RNG

  printf("Running %0.2e lookups with %0.2e gridpoints in a %0.2f MB array...\n", 
      (double) n_lookups, (double) n_grid, (double) n_grid*sizeof(double)/1e6);

  // Populate values for F(x) on grid
  for (i=0; i<n_grid; i++) {
    F_vals[i] = F(i*interval);
  }

  gettimeofday(&start, NULL);

    // Initialize seeds
    // seed = 1000;

#pragma acc data copyin(F_vals[0:n_grid]), copyout(sum)
  {
#pragma acc parallel for reduction(+:sum)
    for (long i=0; i<n_lookups; i++) {

      // Randomly sample a continous value for x
      double x = (double) rn(i);
      // double x = (double) rn(&seed);
      // double x =rn_s(RN_S_SEED, (long) i);
      // double x = (double) rand() / RAND_MAX;


      // Find the indices that bound x on the grid
      long j = x / interval;
      long k = j+1;

      // Calculate interpolation factor
      double f = (k*interval - x) / (k*interval - j*interval);

      // Interpolate and accumulate result
      sum += F_vals[j+1] - f * (F_vals[j+1] - F_vals[j]);
    }
  }

  gettimeofday(&end, NULL);

  wall_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.;

  printf("Result: %0.6f\n", sum / n_lookups);
  printf("Time:   %0.2e s\n", wall_time);
  printf("Rate:   %0.2e lookups/s\n", n_lookups / wall_time);

  return 0;
}
