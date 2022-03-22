/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <gpuRcuda/device_matrix.hpp>

#include "cusolver_utils.h"

template <typename T>
void cusolver_gesvd(
  cusolverDnHandle_t &cusolverH,
  signed char jobu, signed char jobvt,
  int m, int n,
  T* A, int lda,
  T* S,
  T* U, int ldu,
  T* VT, int ldvt,
  T* work, int* lwork,
  T* d_rwork, int* devInfo) {
      throw Rcpp::exception("default gesvd method called in error");
}

template <>
void cusolver_gesvd<double>(
  cusolverDnHandle_t &cusolverH,
  signed char jobu, signed char jobvt,
  int m, int n,
  double* A, int lda,
  double* S,
  double* U, int ldu,
  double* VT, int ldvt,
  double* work, int* lwork,
  double* d_rwork, int* devInfo) {

    /* query working space of SVD */
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, lwork));
    std::cout << "allocating the lwork based on the buffer size " << *lwork << std::endl;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&work), sizeof(double) * (*lwork)));
    std::cout << "invoking the cusolverDnDgesvd API" << std::endl;
    CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, A, lda, S, U,
                                    lda, // ldu
                                    VT,
                                    lda, // ldvt,
                                    work, *lwork, d_rwork, devInfo));
    return;
}

template <typename T>
void cusolverGesvd(SEXP A, SEXP S, SEXP U, SEXP VT, std::string type){

    Rcpp::XPtr<device_matrix<T> > pA(A);
    Rcpp::XPtr<device_matrix<T> > pS(S);
    Rcpp::XPtr<device_matrix<T> > pU(U);
    Rcpp::XPtr<device_matrix<T> > pVT(VT);

    cusolverDnHandle_t cusolverH=NULL;
    
    /* step 1: create cusolver & cublas handle */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    
    //auto gemm_iter = gemm_methods.find(type);
    //cublasStatus_t gemm = gemm_iter->second();
  
    int m = pA->nrow(), n = pA->ncol();
    int lda = m;
    
    int *devInfo = nullptr;

    int lwork = 0; /* size of workspace */
    T* d_work = nullptr;
    T* d_rwork = nullptr;

    std::printf("A = (matlab base-1)\n");
    //print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");
    

    /* compute SVD */
    signed char jobu = 'A';  // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
  
    std::cout << "about to call templated cusolver" << std::endl;
    
    // cublasSgemm(handle, CUBLAS_OPI_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    cusolver_gesvd(cusolverH, 
                        jobu, jobvt, 
                        m, n, 
                        thrust::raw_pointer_cast(pA->getPtr()->data()), lda, 
                        thrust::raw_pointer_cast(pS->getPtr()->data()), 
                        thrust::raw_pointer_cast(pU->getPtr()->data()), lda, 
                        thrust::raw_pointer_cast(pVT->getPtr()->data()), lda,
                        d_work, &lwork, 
                        d_rwork, 
                        devInfo);

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_rwork));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaDeviceReset());
    
    return;
}

// [[Rcpp::export]]
void 
cusolverGesvd(SEXP A, SEXP S, SEXP U, SEXP VT, std::string type, const int type_flag)
{
  std::cout << "entered c++" << std::endl;
  switch(type_flag){
    case 8:
        cusolverGesvd<double>(A, S, U, VT, type);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuRcublas object!");
  }
}
