#' @useDynLib gpuRcublas
#' @importFrom Rcpp evalCpp
#' @import methods
#' @importFrom gpuRcuda cudaMatrix
cublas_gemm <- function(A, B){
# getSlot(A)
  type = "double"
  cat('type A:', type, "\n")
  cat('type B:', type, "\n")
  M=nrow(A)
  N=ncol(B) 
  cat("sizeof(C) = (", M, ",", N, ")\n")

  C <- matrix(0,nrow=M,ncol=N)

  C <- cudaMatrix(C, nrow = M, ncol = N, type = type)
  
  print('output initialized')

  cublasGemm(A@address,B@address,C@address,"double",8L)
    
  print('multiplication done after the call')

# switch(type,
#        float = {cpp_gpuMatrix_gemm(A@address,
#                                    B@address,
#                                    C@address,
#                                    6L)
#        },
#        double = {
#          cpp_gpuMatrix_gemm(A@address,
#                             B@address,
#                             C@address,
#                             8L)
#        },
#        stop("type not implemented")
# )
  
  return(C)
  

}

cusolver_gesvd <- function(A){
  type = "double"
  cat('type A:', type, "\n")
  M=nrow(A)
  N=ncol(A)
  L=M

  cat("sizeof(L) = (", L, ")\n")
  
  U  <- matrix(0,nrow=M,ncol=M);  # m-by-m unitary matrix, left singular vectors
  S  <- matrix(0,nrow=N,ncol=1);    # numerical singular value 
  VT <- matrix(0,nrow=N,ncol=N)   # n-by-n unitary matrix, right singular vectors 

  U  <- cudaMatrix(U, nrow = M, ncol = M, type = type)
  S  <- cudaMatrix(S, nrow = N, ncol = 1, type = type)
  VT <- cudaMatrix(VT, nrow = N, ncol = N, type = type)
  
  print('output initialized')

  cusolverGesvd(A@address,S@address,U@address,VT@address,"double",8L)
    
  print('gesvd done after the call')
  
  return(VT)
}


