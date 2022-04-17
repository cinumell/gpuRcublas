#!/usr/bin/env R
# 4.1
library('Matrix')
library('microbenchmark')
library('gpuR')

mygemm <- function(ncol,nrun) {
  A  <- matrix(rnorm(ncol^2), ncol=ncol)
  B  <- matrix(rnorm(ncol^2), ncol=ncol)
  gA <- gpuMatrix(A,type="double")
  gB <- gpuMatrix(B,type="double")

# C   <- A %*% B
  gC  <- gA%*%gB

# expect_equal(C  ,gC[],tolerance=1e-10,info="gemm elements not equiv")

  r  <- microbenchmark(
    gpuR <- gA %*% gB,
    times = nrun
  )

  return(r)
}

args <- commandArgs(TRUE)
N = as.numeric(args[1])
ncol = 2^N
nrun = 100
r <- mygemm(ncol,nrun)
print(ncol)
print(r)

