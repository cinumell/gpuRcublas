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
  S  <- matrix(0,nrow=N,ncol=1);  # numerical singular value 
  VT <- matrix(0,nrow=N,ncol=N)   # n-by-n unitary matrix, right singular vectors 

  U  <- cudaMatrix(U, type = type)
  S  <- cudaMatrix(S, type = type)
  VT <- cudaMatrix(VT,type = type)
  
  print('output initialized')

  cusolverGesvd(A@address,S@address,U@address,VT@address,"double",8L)
    
  print('gesvd done after the call')
  
  ret_vals <- list("d" = S, "u" = U, "vt" = VT)
  return(ret_vals)
}


cusolver_Xgetrf <- function(A, B){
  type = "double"
  cat('type A:', type, "\n")
  M=nrow(A)
  N=ncol(A)
  L=M

  cat("sizeof(L) = (", L, ")\n")
  
  X   <- matrix(0,nrow=M,ncol=1);  
  LU  <- matrix(0,nrow=L,ncol=M); 
  PIV <- matrix(0,nrow=M,ncol=1);

  X  <- cudaMatrix(X, type = type)
  LU  <- cudaMatrix(LU, type = type)
  PIV <- cudaMatrix(PIV,type = integer)
  
  print('output initialized')

  cusolverXgetrf(A@address,B@address,PIV@address,LU@address,X@address,"double",8L)
    
  print('getrf done after the call')
  
  ret_vals <- list("PIV" = PIV, "LU" = LU, "X" = X)
  return(ret_vals)
}


