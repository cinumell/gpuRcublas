#!/usr/bin/env R
# 4.1

library('devtools')
library('testthat')
.libPaths( c(.libPaths(), "/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas/lib") )
library('gpuRcuda')
library('gpuRcublas')

#A <- matrix(c(1,0,0,0,  # col1
#              0,0,0,2,  # col2
#              0,3,0,0,  # col3 
#              0,0,0,0,  # col4
#              2,0,0,0   # col5
#), nrow=4)

A <- matrix(c(1,4,2,  # col1
              2,5,1   # col2
), nrow=3)

cpu_SVD <- La.svd(A)
print("expected output")
print(cpu_SVD$d)
print(cpu_SVD$u)
print(cpu_SVD$vt)
gpu_SVD <- svd(cudaMatrix(A,type="double"))

print(gpu_SVD$d[])
print(gpu_SVD$u[])
print(gpu_SVD$vt[])
