#include "cuda_utils.hpp"
#include "spmm_coo.hpp"
#include "torch/torch.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace cuspmm {

template <typename T, typename MT, typename AccT>
__global__ void spmmCOOK1(MT aNumRows, MT aNumCols, MT aNumNonZero,
                                    MT *rowIdxs, MT *colIdxs, T* aData, 
                                    MT bNumRows, MT bNumCols, T* bData,
                                    T* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < aNumNonZero) {
        int row = rowIdxs[idx];
        int col = colIdxs[idx];
        float value = aData[idx];

        for (int j = 0; j < bNumCols; j++) {
            atomicAdd(&cData[row * bNumCols + j], value * bData[col * bNumCols + j]);
        }
    }
}

template <typename T, typename AccT>
DenseMatrix<T>* spmmCooDevice(SparseMatrixCOO<T>* a, DenseMatrix<T>* b) {
    const size_t numNonZero = a->numNonZero;

    const size_t BLOCKSIZE = 1024;

    dim3 block(BLOCKSIZE);
    dim3 grid((numNonZero + BLOCKSIZE - 1) / BLOCKSIZE);

    if (!a->onDevice || !b->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    DenseMatrix<T>* c = new DenseMatrix<T>(a->numRows, b->numCols, true);

    spmmCOOK1<T, typename SparseMatrixCOO<T>::metadataType, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->rowIdxs, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );

    return c;
}

template <typename T>
void runEngineCOO(SparseMatrixCOO<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol) {

    // 1. Move to device
    SparseMatrixCOO<T>* da = a->copy2Device();
    DenseMatrix<T>* db = b->copy2Device();

    // 2. Launch kernel
    auto cRes = spmmCooDevice<T, double>(da, db);
    auto cResCpu = cRes->copy2Host();
    cResCpu->save2File("coo_cuda.res");

    // 3. Check result
    auto cResSeq = spmmCooCpu<T, double>(a, b);
    cResSeq->save2File("coo_cpu.res");

    auto denseA = a->toDense();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor taDevice = torch::from_blob(denseA->data, {denseA->numRows, denseA->numCols}, options).clone().cuda();
    torch::Tensor tbDevice = torch::from_blob(b->data, {b->numRows, b->numCols}, options).clone().cuda();
    torch::Tensor tcCpu = torch::from_blob(cResCpu->data, {cResCpu->numRows, cResCpu->numCols}, options).clone();
    torch::Tensor cResTorch = torch::matmul(taDevice, tbDevice).cpu();
    std::cout << "coo allclose: " << torch::allclose(tcCpu, cResTorch, rel_tol, abs_tol) << std::endl;

    auto denseTorch = new DenseMatrix<T>(cResCpu->numRows, cResCpu->numCols, false);
    std::memcpy(denseTorch->data, cResTorch.data_ptr<float>(), denseTorch->numRows * denseTorch->numCols * sizeof(float));
    denseTorch->save2File("coo_torch.res");
}

template void runEngineCOO<float>(SparseMatrixCOO<float> *a, DenseMatrix<float>* b, float abs_tol, double rel_tol);

} // namespace cuspmm