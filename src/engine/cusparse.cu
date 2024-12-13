#include "spmm_cusparse.hpp"
#include "format.hpp"

namespace cuspmm {

template <typename DataT>
DenseMatrix<DataT>* cusparseTest(SparseMatrix<DataT>* a, DenseMatrix<DataT>* b) {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    DenseMatrix<DataT>* c = new DenseMatrix<DataT>(a->numRows, b->numCols, true, ORDERING::ROW_MAJOR);

    if (b->ordering != ORDERING::COL_MAJOR) {
        b->toOrdering(ORDERING::COL_MAJOR);
    }

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

    std::cout << "cusparse prep time (us):" << prepTime.count() << ','
              << "cusparse kernel time (us):" << kernelTime.count() << ','
              << "cusparse epilogue time (us):" << epilogueTime.count() << std::endl;
    
    return c;
}

template DenseMatrix<float>* cusparseTest(SparseMatrix<float>* a, DenseMatrix<float>* b);

}

