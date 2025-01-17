

# moran

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)

<!-- badges: end -->

Trying to generate Moran Eigenvector Maps in Rust using a spatial
weights matrix as input.

## Example

``` r
library(bench)
library(dplyr)
library(moran)

# number of points
n <- 1000

# spatial-weights matrix
W <- matrix(0, ncol = n, nrow = n)

set.seed(1701) # NCC-1701

W[upper.tri(W)] <- runif(n*(n-1)/2, 0, 20)
W <- W + t(W)
W <- round(W, 2)
```

Using `{adespatial}` as a template, this is the basic way to compute a
MEM in R:

``` r
r_mem <- function(x){

  # double-center x
  R = x*0 + rowMeans(x)
  C = t(x*0 + colMeans(x))
  omega <- x - R - C + mean(x)

  # eigendecomposition
  V <- eigen(omega)

  # moran eigenvector map
  V[["vectors"]]

}
```

And a simple benchmark:

``` r
bench::mark(
  r = r_mem(W),
  rust = mem(W),
  check = FALSE,
  relative = TRUE
) |> select(expression:mem_alloc)
#> # A tibble: 2 × 5
#>   expression   min median `itr/sec` mem_alloc
#>   <bch:expr> <dbl>  <dbl>     <dbl>     <dbl>
#> 1 r           10.1   9.49      1         13.7
#> 2 rust         1     1         9.54       1
```
