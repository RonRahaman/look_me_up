#look\_me\_up

## Purpose 

These programs are very simple examples for doing stochastic lookups on a data
array.  The key features are: 

- **Multiple threading APIs:**  The programs demonstrate multiple
  threading APIs, including OpenMP, CUDA, OpenACC, and OCCA.  

- **Easily verifiable result:** The data array is populated such that the
  stochastic lookups will perform a Monte Carlo integration.  For any threading
  implementation, the result should be the same, within statistical error. By
  default, the integral is:

  ![alt text](/img/integral.png "integral")
  

## Usage

After compiling, the executables are run as:
  ```
  main [n_lookups [n_gridpoints]]
  ```
where n_lookups is the total number of lookups to perform; and n_gridpoints are
the number of elements in the data array.  Both parameters are optional and
have default values.  
