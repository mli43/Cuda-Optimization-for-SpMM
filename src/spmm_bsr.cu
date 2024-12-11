#include "cuda_utils.hpp"
#include "torch/torch.h"
#include "spmm_bsr.hpp"
#include "commons.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

namespace cuspmm {

template <typename T, typename MT, typename AccT>
__global__ void spmmBSRK1(MT aNumRows, MT aNumCols, MT aBlockRowSize, MT aBlockColSize,
                                    MT aNumBlocks,
                                    MT *aBlockRowPtrs, MT *aBlockColIdxs, T* aData, 
                                    MT bNumRows, MT bNumCols, T* bData,
                                    T* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    // Every thread block is responsible for one `a` block row
    unsigned int blockRowIdx = blockIdx.x;
    unsigned int inBlockRow = threadIdx.x;
    unsigned int inBlockCol = threadIdx.y;
    AccT accumulator = 0.f;

    unsigned int blockRowStartIdx = aBlockRowPtrs[blockRowIdx];
    unsigned int blockRowEndIdx = aBlockRowPtrs[blockRowIdx + 1];

    const unsigned int aDenseRowBase = blockRowIdx * aBlockRowSize;
    for (unsigned aBlockIdx = blockRowStartIdx; aBlockIdx < blockRowEndIdx; aBlockIdx++) {
        unsigned int aBlockCol = aBlockColIdxs[aBlockIdx];
        const unsigned int aDenseColBase = aBlockCol * aBlockColSize;
        T* aBlockData = aData + aBlockIdx * aBlockRowSize * aBlockColSize;

        T aDataElement = aBlockData[RowMjIdx(inBlockRow, inBlockCol, aBlockColSize)];
        unsigned int ar = aDenseRowBase + inBlockRow;
        unsigned int ac = aDenseColBase + inBlockCol;

        for (unsigned bc = 0; bc < bNumCols; bc++) {
            // ! This can be improved. Accumulate locally
            atomicAdd(&cData[RowMjIdx(ar, bc, bNumCols)], aDataElement * bData[RowMjIdx(ac, bc, bNumCols)]);
        }
    }
}

template <typename T, typename AccT>
DenseMatrix<T>* spmmBsrDevice(SparseMatrixBSR<T>* a, DenseMatrix<T>* b) {
    size_t rows = a->numCols, cols = b->numCols;

    // (y, x)
    dim3 block(a->blockColSize, a->blockRowSize);
    dim3 grid(a->numBlockRows);

    if (!a->onDevice || !b->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    DenseMatrix<T>* c = new DenseMatrix<T>(a->numRows, b->numCols, true);

    spmmBSRK1<T, typename SparseMatrixBSR<T>::metadataType, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->blockRowSize, a->blockColSize,
        a->numBlockRows, a->blockRowPtrs, a->blockColIdxs,
        a->data, b->numRows, b->numCols, b->data, 
        c->data
    );

    return c;
}

template <typename T>
void runEngineBSR(SparseMatrixBSR<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol) {

    // 1. Move to device
    SparseMatrixBSR<T>* da = a->copy2Device();
    DenseMatrix<T>* db = b->copy2Device();

    // 2. Launch kernel
    auto cRes = spmmBsrDevice<T, double>(da, db);
    auto cResCpu = cRes->copy2Host();
    cResCpu->save2File("bsr_cuda.res");

    // 3. Check result
    auto cResSeq = spmmBsrCpu<T, double>(a, b);
    cResSeq->save2File("bsr_cpu.res");

    auto denseA = a->toDense();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor taDevice = torch::from_blob(denseA->data, {denseA->numRows, denseA->numCols}, options).clone().cuda();
    torch::Tensor tbDevice = torch::from_blob(b->data, {b->numRows, b->numCols}, options).clone().cuda();
    torch::Tensor tcCpu = torch::from_blob(cResCpu->data, {cResCpu->numRows, cResCpu->numCols}, options).clone();
    torch::Tensor cResTorch = torch::matmul(taDevice, tbDevice).cpu();
    std::cout << "bsr allclose: " << torch::allclose(tcCpu, cResTorch, rel_tol, abs_tol) << std::endl;

    auto denseTorch = new DenseMatrix<T>(cResCpu->numRows, cResCpu->numCols, false);
    std::memcpy(denseTorch->data, cResTorch.data_ptr<float>(), denseTorch->numRows * denseTorch->numCols * sizeof(float));
    denseTorch->save2File("bsr_torch.res");
}

template void runEngineBSR<float>(SparseMatrixBSR<float> *a, DenseMatrix<float>* b, float abs_tol, double rel_tol);

} // namespace cuspmm