#!/usr/bin/env R
# 4.1
library('devtools')
library('testthat')
mypath=file.path(getwd(),"/lib")
.libPaths( c(.libPaths(), mypath) )
library('gpuRcuda')
library('gpuRcublas')
library('microbenchmark')
devtools::document()

mysvd <- function(ncol,nrun) {
  A  <- matrix(rnorm(ncol^2), ncol=ncol)

  cpu <- La.svd(A)
  gpu <- gpusvd(cudaMatrix(A))

  cpuA <- cpu$u %*% diag(cpu$d) %*% cpu$vt
  gpuA <- gpu$u %*% cudaMatrix(diag(drop(gpu$d[]))) %*% gpu$vt
  expect_equal(cpuA  ,A,tolerance=1e-10,info="cpu svd elements not equiv")
  expect_equal(gpuA[],A,tolerance=1e-10,info="gpu svd elements not equiv")

  r  <- microbenchmark(
    cpu <- La.svd(A),
    gpu <- gpusvd(cudaMatrix(A)),
    times = nrun
  )

  return(r)
}

for ( N in 1:15) {
  ncol = 2^N
  nrun = 10
  if ( N <= 10 ) {
      nrun = 100
  }
  r <- mysvd(ncol,nrun)
  print(ncol)
  print(r)
}

