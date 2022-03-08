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

