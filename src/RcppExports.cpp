// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// cublasGemm
void cublasGemm(SEXP A, SEXP B, SEXP C, std::string type, const int type_flag);
RcppExport SEXP _gpuRcublas_cublasGemm(SEXP ASEXP, SEXP BSEXP, SEXP CSEXP, SEXP typeSEXP, SEXP type_flagSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< SEXP >::type B(BSEXP);
    Rcpp::traits::input_parameter< SEXP >::type C(CSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< const int >::type type_flag(type_flagSEXP);
    cublasGemm(A, B, C, type, type_flag);
    return R_NilValue;
END_RCPP
}
// cusolverXgetrf
void cusolverXgetrf(SEXP A, SEXP PIV, SEXP LU, SEXP PIV_FLAG, std::string type, const int type_flag);
RcppExport SEXP _gpuRcublas_cusolverXgetrf(SEXP ASEXP, SEXP PIVSEXP, SEXP LUSEXP, SEXP PIV_FLAGSEXP, SEXP typeSEXP, SEXP type_flagSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< SEXP >::type PIV(PIVSEXP);
    Rcpp::traits::input_parameter< SEXP >::type LU(LUSEXP);
    Rcpp::traits::input_parameter< SEXP >::type PIV_FLAG(PIV_FLAGSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< const int >::type type_flag(type_flagSEXP);
    cusolverXgetrf(A, PIV, LU, PIV_FLAG, type, type_flag);
    return R_NilValue;
END_RCPP
}
// cusolverGesvd
void cusolverGesvd(SEXP A, SEXP S, SEXP U, SEXP VT, std::string type, const int type_flag);
RcppExport SEXP _gpuRcublas_cusolverGesvd(SEXP ASEXP, SEXP SSEXP, SEXP USEXP, SEXP VTSEXP, SEXP typeSEXP, SEXP type_flagSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< SEXP >::type S(SSEXP);
    Rcpp::traits::input_parameter< SEXP >::type U(USEXP);
    Rcpp::traits::input_parameter< SEXP >::type VT(VTSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< const int >::type type_flag(type_flagSEXP);
    cusolverGesvd(A, S, U, VT, type, type_flag);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gpuRcublas_cublasGemm", (DL_FUNC) &_gpuRcublas_cublasGemm, 5},
    {"_gpuRcublas_cusolverXgetrf", (DL_FUNC) &_gpuRcublas_cusolverXgetrf, 6},
    {"_gpuRcublas_cusolverGesvd", (DL_FUNC) &_gpuRcublas_cusolverGesvd, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_gpuRcublas(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
