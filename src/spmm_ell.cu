#include "cuda_utils.hpp"
#include "spmm_ell.hpp"
#include "torch/torch.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <chrono>

namespace cuspmm {

template <typename T, typename MT, typename AccT>
__global__ void spmmELLK1(MT aNumRows, MT aNumCols, MT aNumNonZero, MT aMaxRowNnz,
                                    MT *colIdxs, T* aData, 
                                    MT bNumRows, MT bNumCols, T* bData,
                                    T* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t numValues = aNumRows * aMaxRowNnz;

    if (idx < numValues) {
        int row = idx / aMaxRowNnz;
        int col = colIdxs[idx];
        float value = aData[idx];
        

        if (col >= 0) {
            for (int j = 0; j < bNumCols; j++) {
                atomicAdd(&cData[row * bNumCols + j], value * bData[col * bNumCols + j]); 
            }
        }
    }
}

template <typename T, typename AccT>
DenseMatrix<T>* spmmEllDevice(SparseMatrixELL<T>* a, DenseMatrix<T>* b) {
    const size_t numValues = a->numRows * a->maxRowNnz;

    const size_t BLOCKSIZE = 1024;

    dim3 block(BLOCKSIZE);
    dim3 grid((numValues + BLOCKSIZE - 1) / BLOCKSIZE);

    if (!a->onDevice || !b->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    DenseMatrix<T>* c = new DenseMatrix<T>(a->numRows, b->numCols, true);

    spmmELLK1<T, typename SparseMatrixELL<T>::metadataType, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->maxRowNnz, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );

    return c;
}

template <typename T>
void runEngineELL(SparseMatrixELL<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol) {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Move to device
    SparseMatrixELL<T>* da = a->copy2Device();
    DenseMatrix<T>* db = b->copy2Device();
    auto copy_to_device_end = std::chrono::high_resolution_clock::now();

    // 2. Launch kernel
    auto cRes = spmmEllDevice<T, double>(da, db);
    auto kernel_end = std::chrono::high_resolution_clock::now();

    auto cResCpu = cRes->copy2Host();
    auto copy_to_host_end = std::chrono::high_resolution_clock::now();

    // 3. Check result
    auto cResSeq = spmmEllCpu<T, double>(a, b);
    auto seq_end = std::chrono::high_resolution_clock::now();

    // 4. Report time 
    auto copy2DeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_device_end - start);
    auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - copy_to_device_end);
    auto copy2HostTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_host_end - kernel_end);
    auto seqTime = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - copy_to_host_end);

    std::cout << "copy2DeviceTime (us):" << copy2DeviceTime.count() << ','
              << "kernelTime (us):" << kernelTime.count() << ','
              << "copy2HostTime (us):" << copy2HostTime.count() << ','
              << "seqTime (us):" << seqTime.count() << '\n';

    cResCpu->save2File("ell_cuda.res");
    cResSeq->save2File("ell_cpu.res");

    auto denseA = a->toDense();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor taDevice = torch::from_blob(denseA->data, {denseA->numRows, denseA->numCols}, options).clone().cuda();
    torch::Tensor tbDevice = torch::from_blob(b->data, {b->numRows, b->numCols}, options).clone().cuda();
    torch::Tensor tcCpu = torch::from_blob(cResCpu->data, {cResCpu->numRows, cResCpu->numCols}, options).clone();
    torch::Tensor cResTorch = torch::matmul(taDevice, tbDevice).cpu();
    std::cout << "ell allclose: " << torch::allclose(tcCpu, cResTorch, rel_tol, abs_tol) << std::endl;

    auto denseTorch = new DenseMatrix<T>(cResCpu->numRows, cResCpu->numCols, false);
    std::memcpy(denseTorch->data, cResTorch.data_ptr<float>(), denseTorch->numRows * denseTorch->numCols * sizeof(float));
    denseTorch->save2File("ell_torch.res");
}

template void runEngineELL<float>(SparseMatrixELL<float> *a, DenseMatrix<float>* b, float abs_tol, double rel_tol);

} // namespace cuspmm
