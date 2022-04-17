#!/usr/bin/env R
# 4.1

library('devtools')
library('testthat')
.libPaths( c(.libPaths(), "/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas/lib") )
library('gpuRcuda')
library('gpuRcublas')

v <- sqrt(2)/2
U <- matrix(c(v,-v,v,v), nrow=2)
VT <- t(U)
S <- matrix(c(3,0,0,1/3), nrow=2)
A <- U %*% S %*% VT

cpu_SVD <- La.svd(A)
gpu_SVD <- gpusvd(cudaMatrix(A,type="double"))

ORDER <- 2^10

A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
gpuA <- cudaMatrix(A, type="double")
M = nrow(gpuA)
N = ncol(gpuA)
type=typeof(gpuA)
cat("type of gpuA:", type, "\n")
cat("dimensions of gpuA:", M, N, "\n")

cpu_SVD <- La.svd(A)
gpu_SVD <- gpusvd(gpuA)

cat("type of gpu_SVD$d:", typeof(gpu_SVD$d), "\n")

expect_equal(cpu_SVD$d, as.vector(gpu_SVD$d[]), tolerance=1e-07, 
               info="matrix elements not equivalent")  
 
cat("type of gpu_SVD$u:", typeof(gpu_SVD$u), "\n")

expect_equal(cpu_SVD$u, gpu_SVD$u[], tolerance=1e-07, 
               info="matrix elements not equivalent")  

cat("type of gpu_SVD$vt:", typeof(gpu_SVD$vt), "\n")

expect_equal(cpu_SVD$vt, gpu_SVD$vt[], tolerance=1e-07, 
               info="matrix elements not equivalent")  
cat("End of the program", "\n")
