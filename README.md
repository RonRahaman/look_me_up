#Look\_Me\_Up

## Purpose 

These programs are very simple benchmarks for doing stochastic lookups on data
arrays.  The key features are: 

- **Simple tool for learning threading APIs:**  We hope the program is simple
  enough to be a quick and easy introduction to new threading APIs, such as
  OpenACC and OCCA.  

- **Example for stochastic lookuups:** Most threading tutorials focus on
  linear algebra applications.  There are few examples for stochastic
  lookups, which can pose interesting challenges.

- **Easily verifiable result:** The data arrays are populated such that the
  lookups perform a Monte Carlo integration.  The result should be
  statistically indistinguishable for any threading implementation. By default,
  the integral is:

  ![alt text](/img/integral.png "integral")
  

## Usage

After compiling, the executables are run as:
  ```
  main [n_lookups [n_gridpoints]]
  ```
Both parameters are optional and have default values.  
