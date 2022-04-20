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

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <gpuRcuda/device_matrix.hpp>
#include <gpuRcuda/host_matrix.hpp>
#include "cusolver_utils.h"

#if 0
void print_LU_matrix(const int64_t &m, const int64_t &n, double *A, const int64_t &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

void print_pivoting_seq(int *Ipiv, int &m) {
    for (int j = 0; j < m; j++) {
        std::printf("Ipiv(%d) = %lu\n", j + 1, Ipiv[j]);
    }
    return;
}
#endif

template <typename T>
void cusolver_Xgetrf(
    cusolverDnHandle_t cusolverH,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    T *A,
    int64_t lda,
    int64_t *ipiv,
    cudaDataType computeType,
    T *d_work,
    size_t *d_lwork,
    T *h_work,
    size_t *h_lwork,
    int *d_info, const int pivot_on ) {
      throw Rcpp::exception("default getrf method called in error");
}

template <>
void cusolver_Xgetrf<double>(
    cusolverDnHandle_t cusolverH,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    double *A,
    int64_t lda,
    int64_t *ipiv,
    cudaDataType computeType,
    double *d_work,
    size_t *d_lwork,
    double *h_work,
    size_t *h_lwork,
    int *d_info, const int pivot_on ) {

    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, n, dataTypeA, (void *)A, lda, ipiv, 
                                        computeType, (void *)d_work, *d_lwork, (void *)h_work, *h_lwork, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, n, dataTypeA, (void *)A, lda, nullptr, 
                                        computeType,(void *)d_work, *d_lwork, (void *)h_work, *h_lwork, d_info));
    }
    return;
}


template <typename T>
void cusolverXgetrf(SEXP A, SEXP PIV, SEXP LU, SEXP PIV_FLAG, std::string type){

    Rcpp::XPtr<device_matrix<T> > pA(A);
    Rcpp::XPtr<device_matrix<T> > pLU(LU);
    Rcpp::XPtr<host_matrix<int>   > pIV(PIV);
    int pivot_on = Rcpp::as<int >(PIV_FLAG); 
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    using data_type=T;

    const int64_t m = pA->nrow();
    const int64_t lda = m;
    const int64_t n = pA->ncol();

    int info = 0;
    int *d_info = nullptr;     /* error info */

    size_t d_lwork = 0;     /* size of workspace */
    void *d_work = nullptr; /* device workspace for getrf */
    size_t h_lwork = 0;     /* size of workspace */
    void *h_work = nullptr; /* host workspace for getrf */
    size_t d_lIpiv= 0;      /* size of workspace */
    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    const int algo = 0;
    //std::vector<data_type> LU_mat(lda*m,0);

//  if (pivot_on) {
//      std::printf("pivot is on : compute P*A = L*U \n");
//  } else {
//      std::printf("pivot is off: compute A = L*U (not numerically stable)\n");
//  }

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* Create advanced params */
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    if (algo == 0) {
//      std::printf("Using New Algo\n");
        CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
    } else {
//      std::printf("Using Legacy Algo\n");
        CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
    }
    /* step 2: query working space of getrf */
    d_lIpiv=pIV->ncol();
    std::printf("pivot sequence length is %lu\n", d_lIpiv);
    CUSOLVER_CHECK(
        cusolverDnXgetrf_bufferSize(cusolverH, params, m, m, traits<data_type>::cuda_data_type, thrust::raw_pointer_cast(pA->getPtr()->data()),
                                    lda, traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * d_lIpiv));
    /* step 3: LU factorization */
    // XXX pivoting is broken
    cusolver_Xgetrf(cusolverH, params, m, m, traits<data_type>::cuda_data_type,thrust::raw_pointer_cast(pA->getPtr()->data()), lda, 
                     d_Ipiv, traits<data_type>::cuda_data_type,
                     (double *)d_work, &d_lwork, (double *)h_work, &h_lwork, d_info, pivot_on);
    
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(pLU->getPtr()->data()), thrust::raw_pointer_cast(pA->getPtr()->data()), 
                                sizeof(data_type) * lda * m, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(pIV->getPtr()->data()), reinterpret_cast<int *>(d_Ipiv), 
                                sizeof(int) * d_lIpiv, cudaMemcpyDeviceToHost, stream));
#if 0
    CUDA_CHECK(cudaMemcpyAsync(LU_mat.data(), reinterpret_cast<data_type *>(thrust::raw_pointer_cast(pA->getPtr()->data())),
                                sizeof(data_type) * lda * m, cudaMemcpyDeviceToHost, stream));
#endif
    CUDA_CHECK(cudaStreamSynchronize(stream));
#if 0
    print_pivoting_seq(thrust::raw_pointer_cast(pIV->getPtr()->data()), d_lIpiv);
    print_LU_matrix(m,n,LU_mat.data(), lda);    
#endif

    if (info !=0) {
        std::printf("after Xgetrf: info = %d\n", info);
        std::printf("=====\n");
    }

    /* free resources */
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_Ipiv));

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    return;
}

// [[Rcpp::export]]
void
cusolverXgetrf(SEXP A, SEXP PIV, SEXP LU, SEXP PIV_FLAG, std::string type, const int type_flag)
{
  //std::cout << "entered c++" << std::endl;
  switch(type_flag){
    case 8:
        cusolverXgetrf<double>(A, PIV, LU, PIV_FLAG, type);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuRcublas object!");
  }
}
