#!/usr/bin/env R
# 4.1

library('devtools')
.libPaths( c(.libPaths(), "/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas/lib") )
library('gpuRcuda')
library('gpuRcublas')


ORDER <- 2^10

A <- matrix(rnorm(ORDER^2), nrow=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER)

gpuA <- cudaMatrix(A, type="double")
M = nrow(gpuA)
N = ncol(gpuA)
type=typeof(gpuA)
cat("type of gpuA:", type, "\n")
cat("dimensions of gpuA:", M, N, "\n")
gpuB <- cudaMatrix(B, nrow=M, ncol=N, type=type)

cat("type of gpuB:", typeof(gpuB), "\n")
#C = A %*% B
gpuC <- gpuA %*% gpuB

all.equal(C,gpuC[])
