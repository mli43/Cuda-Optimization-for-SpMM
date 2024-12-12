#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/sparse_csr.hpp"
#include "torch/torch.h"
#include "spmm_csr.hpp"
#include "spmm_cusparse.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace cuspmm {

template <typename T, typename MT, typename AccT>
__global__ void spmmCSRK1(MT aNumRows, MT aNumCols, MT aNumNonZero,
                                    MT *rowPtrs, MT *colIdxs, T* aData, 
                                    MT bNumRows, MT bNumCols, T* bData,
                                    T* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c < bNumCols and r < aNumRows) {
        AccT acc = .0f;
        unsigned int row_start = rowPtrs[r];
        unsigned int row_end = rowPtrs[r + 1];

        for (unsigned int i = row_start; i < row_end; i++) {
            unsigned int c_idx = colIdxs[i];
            T aValue = aData[i];
            acc += aValue * bData[c_idx * bNumCols + c];
        }
        cData[r * bNumCols + c] = acc;
    }
}

template <typename T, typename AccT>
DenseMatrix<T>* spmmCsrDevice(SparseMatrixCSR<T>* a, DenseMatrix<T>* b) {
    size_t rows = a->numCols, cols = b->numCols;

    const size_t BLOCKSIZE = 32;

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((cols + BLOCKSIZE - 1) / BLOCKSIZE, (rows + BLOCKSIZE - 1) / BLOCKSIZE);

    if (!a->onDevice || !b->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    DenseMatrix<T>* c = new DenseMatrix<T>(a->numRows, b->numCols, true);

    spmmCSRK1<T, typename SparseMatrixCSR<T>::metadataType, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->rowPtrs, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );

    return c;
}

template <typename T>
void spmmCsrCuSparse(SparseMatrixCSR<T>* a, DenseMatrix<T> b) {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    // FIXME: Only supports float right not!
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, a->numRows, a->numCols, a->numNonZero, a->rowPtrs, a->colIdxs, a->data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F));
    
    throw std::runtime_error("Not implemented");

}

template <typename T>
void runEngineCSR(SparseMatrixCSR<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol) {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Move to device
    SparseMatrixCSR<T>* da = a->copy2Device();
    DenseMatrix<T>* db = b->copy2Device();
    auto copy_to_device_end = std::chrono::high_resolution_clock::now();

    // 2. Launch kernel
    auto cRes = spmmCsrDevice<T, double>(da, db);
    auto kernel_end = std::chrono::high_resolution_clock::now();

    auto cResCpu = cRes->copy2Host();
    auto copy_to_host_end = std::chrono::high_resolution_clock::now();

    // 3. Check result
    auto cResSeq = spmmCsrCpu<T, double>(a, b);
    auto seq_end = std::chrono::high_resolution_clock::now();

    // 4. Report time 
    auto copy2DeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_device_end - start);
    auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - copy_to_device_end);
    auto copy2HostTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_host_end - kernel_end);
    auto parallelTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_host_end - start);
    auto seqTime = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - copy_to_host_end);

    std::cout << "copy2DeviceTime (us):" << copy2DeviceTime.count() << ','
              << "kernelTime (us):" << kernelTime.count() << ','
              << "copy2HostTime (us):" << copy2HostTime.count() << ','
              << "parallelTime (us):" << parallelTime.count() << ','
              << "seqTime (us):" << seqTime.count() << '\n';

    cResCpu->save2File("csr_cuda.res");
    cResSeq->save2File("csr_cpu.res");

    auto denseA = a->toDense();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor taDevice = torch::from_blob(denseA->data, {denseA->numRows, denseA->numCols}, options).clone().cuda();
    torch::Tensor tbDevice = torch::from_blob(b->data, {b->numRows, b->numCols}, options).clone().cuda();
    torch::Tensor tcCpu = torch::from_blob(cResCpu->data, {cResCpu->numRows, cResCpu->numCols}, options).clone();
    torch::Tensor cResTorch = torch::matmul(taDevice, tbDevice).cpu();
    std::cout << "csr allclose: " << torch::allclose(tcCpu, cResTorch, rel_tol, abs_tol) << std::endl;

    auto denseTorch = new DenseMatrix<T>(cResCpu->numRows, cResCpu->numCols, false);
    std::memcpy(denseTorch->data, cResTorch.data_ptr<float>(), denseTorch->numRows * denseTorch->numCols * sizeof(float));
    denseTorch->save2File("csr_torch.res");
}

template void runEngineCSR<float>(SparseMatrixCSR<float> *a, DenseMatrix<float>* b, float abs_tol, double rel_tol);

} // namespace cuspmm
