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

mygemm <- function(ncol,nrun) {
  A  <- matrix(rnorm(ncol^2), ncol=ncol)
  B  <- matrix(rnorm(ncol^2), ncol=ncol)
  gA <- cudaMatrix(A)
  gB <- cudaMatrix(B)

  C   <- A %*% B
  gC  <- gA%*%gB

# expect_equal(C  ,gC[],tolerance=1e-10,info="gemm elements not equiv")

  r  <- microbenchmark(
    cpu <- A %*% B,
    gpu <- gA %*% gB,
    times = nrun
  )

  return(r)
}

#for ( N in 8) {
#  ncol = 2^N
#  nrun = 100
## if ( N <= 10 ) {
##     nrun = 100
## }
#  r <- mygemm(ncol,nrun)
#  print(ncol)
#  print(r)
#}
#

args <- commandArgs(TRUE)
N = as.numeric(args[1])
ncol = 2^N
nrun = 100
r <- mygemm(ncol,nrun)
print(ncol)
print(r)
