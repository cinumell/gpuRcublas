#!/usr/bin/env R
# 4.1

mypath=file.path(getwd(), "lib")
#ENV BACKEND=CUDA
dir.create(mypath, showWarnings = FALSE)
withr::with_libpaths(new=mypath,devtools::install_github('cinumell/thrust'))
withr::with_libpaths(new=mypath,devtools::install_github('RcppCore/Rcpp'))
withr::with_libpaths(new=mypath,devtools::install_github('cinumell/gpuRcuda'))
.libPaths( c(.libPaths(), mypath))
library('gpuRcuda')
withr::with_libpaths(new=mypath,devtools::install_github('cinumell/gpuRcublas_new'))
