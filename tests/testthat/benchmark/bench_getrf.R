#!/usr/bin/env R
# 4.1
library('devtools')
library('testthat')
.libPaths( 
  c(
    .libPaths(), 
    "/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas_new_benchmark/lib"
  ) 
)
library('gpuRcuda')
library('gpuRcublas')
library('microbenchmark')
library('Matrix')
devtools::document()

mylu <- function(ncol,nrun) {
  A  <- matrix(rnorm(ncol^2), ncol=ncol)

# cpu <- expand(lu(A))
  # The matrices in the gpu obj. are mostly on cpu
  gpu <- gpulu(cudaMatrix(A),PIV_FLAG=1)

# cpuA <- t(cpu$P) %*% cpu$L %*% cpu$U
  gpuA <- t(gpu$P) %*% gpu$L %*% gpu$U
  expect_equal(cpuA  ,A,tolerance=1e-10,info="cpu LU elements not equiv")
  expect_equal(gpuA[],A,tolerance=1e-10,info="gpu LU elements not equiv")

  r  <- microbenchmark(
#   cpu <- expand(lu(A)),
    gpu <- gpulu(cudaMatrix(A),PIV_FLAG=1),
    times = nrun
  )

  return(r)
}

for ( N in 1:14) {
  ncol = 2^N
  nrun = 100
# if ( N <= 10 ) {
#     nrun = 100
# }
  r <- mylu(ncol,nrun)
  print(ncol)
  print(r)
}

