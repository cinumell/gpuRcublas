# gpuRcublas: The Simple CUDA BLAS Interface for R
[![Travis-CI Build Status](https://travis-ci.org/gpuRcore/gpuRcublas.png?branch=master)](https://travis-ci.org/gpuRcore/gpuRcublas)



Welcome to gpuRcublas!  This package is designed to be an extension upon the more general [gpuRcuda](https://github.com/cinumell/gpuRcuda) and 
[thrust](https://github.com/cinumell/thrust) packages.
Essentially,this package provides the linear algebra routines not implemented in [gpuRcuda](https://github.com/cinumell/gpuRcuda). 
The key aspect of this package is to allow the user to use a CUDA backend while also leveraging the cublas library.

The syntax is designed to be identical to [gpuR](https://github.com/cdeterman/gpuR)

Pre-built environment setup:
The sytem/docker should have the modules loaded with the libraries:
1. R/4.1.0-BLAS
2. cuda/11.2.0

Installation:
Run the install script file --> Rscript install.R

The project was accepted in Birds of Feathers PEARC'22 in Boston. The submitted short paper can be found [here](https://github.com/cinumell/gpuRcublas/blob/main/BoF.pdf).
