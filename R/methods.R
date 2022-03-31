#' @import methods

#' @title cuBLAS Matrix Multiplication
#' @description Multiply two gpuRcude objects, if they are conformable.
#' @param x A gpuRcuda object
#' @param y A gpuRcuda object
#' @docType methods
#' @rdname grapes-times-grapes-methods
#' @author Charles Determan Jr.
#' @importClassesFrom gpuRcuda cudaMatrix
#' @export
setMethod("%*%", signature(x="cudaMatrix", y = "cudaMatrix"),
          function(x,y)
          {
            if( dim(x)[2] != dim(y)[1]){
              stop("Non-conformant matrices")
            }
            return(cublas_gemm(x, y))
          },
          valueClass = "cudaMatrix"
)


#' @title cuSOLVER svd of a gpuRcudaMatrix
#' @description return the singular value decomposition of a gpuRcudaMatrix
#' @param x A gpuRcudaMatrix object
#' @docType methods
#' @rdname cusolver-methods
#' @author Chaitanya Inumella.
#' @importClassesFrom gpuRcuda cudaMatrix
#' @export
setMethod('svd', signature(x="gpuRcudaMatrix"),
           function(x) 
           {
                return(cusolver_gesvd(x))
           },
           valueClass = "cudaMatrix"                  
)


#' @title cuSOLVER getrf of a gpuRcudaMatrices
#' @description return the LU factorization of the gpuRcudaMatrices
#' @param x A gpuRcudaMatrix object
#' @param y A gpuRcudaMatrix object
#' @docType methods
#' @rdname cusolver-methods
#' @author Chaitanya Inumella.
#' @importClassesFrom gpuRcuda cudaMatrix
#' @aliases lud,cudaMatrix
#' @export
setMethod('lud', signature(x="gpuRcudaMatrix", y="gpuRcudaMatrix"),
          function(x,y)
          {
                return(cusolver_Xgetrf(x,y))
          },
          valueClass = "cudaMatrix"
)
