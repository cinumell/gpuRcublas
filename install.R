#!/usr/bin/env R
# 4.1

mypath="/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas_new_benchmark/lib"

#ENV BACKEND=CUDA
library('devtools')
withr::with_libpaths(new=mypath,devtools::install_github('wrathematics/thrust'))
withr::with_libpaths(new=mypath,devtools::install_github('RcppCore/Rcpp'))
withr::with_libpaths(new=mypath,devtools::install_github('cinumell/gpuRcuda',ref="test_fix"))
.libPaths( c(.libPaths(), mypath))
library('gpuRcuda')
withr::with_libpaths(new=mypath,devtools::install_github('cinumell/gpuRcublas_new', force=TRUE))
