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
  lookups perform a Monte Carlo integration.  By default, the integral is:
  ![alt text](/img/integral.png "integral")
  The result should be statistically indistinguishable for any threading implementation.

- **Some semblance to cross-section lookups:** The operations loosely resemble
  microscopic cross-section lookups in Monte Carlo neutron transport, which is
  our target applicaiton.  This algorithm has been abstracted from [XSBench](https://github.com/ANL-CESAR/XSBench), 
  which was abstracted from [OpenMC](https://github.com/mit-crpg/openmc).

## Usage

After compiling, the executables are run as:
  ```
  main [n_lookups [n_gridpoints]]
  ```
Both parameters are optional and have default values.  
