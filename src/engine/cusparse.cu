#include "commons.hpp"
#include "formats/dense.hpp"
#include "spmm_cusparse.hpp"
#include "format.hpp"
#include <cstdint>

namespace cuspmm {

template <typename DT, typename MT>
DenseMatrix<DT, MT>* cusparseTest(SparseMatrix<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* c, long& pro, long& kernel, long& epi) {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    // if (b->ordering != ORDERING::COL_MAJOR) {
    //     b->toOrdering(ORDERING::COL_MAJOR);
    // }

    auto t1 = std::chrono::high_resolution_clock::now();

    CHECK_CUSPARSE(cusparseCreate(&handle));
    a->setCusparseSpMatDesc(&matA);
    b->setCusparseDnMatDesc(&matB);
    c->setCusparseDnMatDesc(&matC);

    float alpha = 1.0f, beta = 0.f;
    void* dBuffer = nullptr;
    size_t buffersize = 0;
    cusparseSpMMAlg_t alg = a->getCusparseAlg();
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, &buffersize));
    cudaCheckError(cudaMalloc(&dBuffer, buffersize));
    auto t2 = std::chrono::high_resolution_clock::now();
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, dBuffer));
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    cudaCheckError(cudaFree(dBuffer));
    auto t4 = std::chrono::high_resolution_clock::now();

    auto prepTime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    auto epilogueTime = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);

    // std::cout << __func__ << ' '
    //           << "cusparse prep time (us):" << prepTime.count() << ','
    //           << "cusparse kernel time (us):" << kernelTime.count() << ','
    //           << "cusparse epilogue time (us):" << epilogueTime.count() << std::endl;
    pro = prepTime.count();
    kernel = kernelTime.count();
    epi = epilogueTime.count();
    
    return c;
}

template DenseMatrix<float, uint32_t>* cusparseTest(SparseMatrix<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c, long& pro, long& kernel, long& epi);
template DenseMatrix<double, uint32_t>* cusparseTest(SparseMatrix<double, uint32_t>* a, DenseMatrix<double, uint32_t>* b, DenseMatrix<double, uint32_t>* c, long& pro, long& kernel, long& epi);

}

