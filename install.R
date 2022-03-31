#!/usr/bin/env R
# 4.1

#ENV BACKEND=CUDA
library('devtools')
.libPaths( c(.libPaths(), "/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas_new/lib"))
withr::with_libpaths(new="/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas_new/lib",devtools::install_github('wrathematics/thrust'))
withr::with_libpaths(new="/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas_new/lib",devtools::install_github('RcppCore/Rcpp'))
withr::with_libpaths(new="/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas_new/lib",devtools::install_github('cinumell/gpuRcuda',ref="test_fix"))
library('gpuRcuda')
withr::with_libpaths(new="/data/sse/scientific-software/cinumell/rmagma_test/gpuRcublas_new/lib",devtools::install_github('cinumell/gpuRcublas_new', force=TRUE))
