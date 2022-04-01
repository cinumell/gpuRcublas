#' @import methods

#' @title cuBLAS Matrix Multiplication
#' @description Multiply two gpuRcuda objects, if they are conformable.
#' @param x a gpuRcuda object
#' @param y a gpuRcuda object
#' @docType methods
#' @rdname grapes-times-grapes-methods
#' @author Charles Determan Jr.
#' @importClassesFrom gpuRcuda cudaMatrix
#' @export
setMethod("%*%", signature(x="cudaMatrix", y= "cudaMatrix"),
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
#' @param x a gpuRcudaMatrix object
#' @docType methods
#' @rdname cusolver-methods
#' @author Chaitanya Inumella.
#' @importClassesFrom gpuRcuda cudaMatrix
#' @export
setGeneric("gpusvd", function(x, ...){
               standardGeneric("gpusvd")
})
setMethod('gpusvd', signature(x="gpuRcudaMatrix"),
           function(x,...) 
           {
                return(cusolver_gesvd(x))
           },
           valueClass = "cudaMatrix"                  
)


#' @title cuSOLVER LU of a gpuRcudaMatrix
#' @description return the LU decomposition of a gpuRcudaMatrix
#' @param x a gpuRcudaMatrix object
#' @docType methods
#' @rdname cusolver-methods
#' @author Chaitanya Inumella.
#' @importClassesFrom gpuRcuda cudaMatrix
#' @export
setGeneric("gpulu", function(x, PIV_FLAG=1, ...){
               standardGeneric("gpulu")
})
setMethod("gpulu", signature(x="gpuRcudaMatrix"),
           function(x, PIV_FLAG=1, ...) 
           {
                return(cusolver_xgetrf(x))
           },
           valueClass = "cudaMatrix"                  
)


