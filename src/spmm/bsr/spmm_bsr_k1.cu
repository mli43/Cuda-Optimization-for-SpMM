#include "cuda_utils.hpp"
#include "commons.hpp"
#include "formats/sparse_bsr.hpp"
#include <cstdint>

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
__global__ void spmmBSRK1(MT aNumRows, MT aNumCols, MT aBlockRowSize, MT aBlockColSize,
                                    MT aNumBlocks,
                                    MT *aBlockRowPtrs, MT *aBlockColIdxs, DT* aData, 
                                    MT bNumRows, MT bNumCols, DT* bData,
                                    DT* cData) {
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
        DT* aBlockData = aData + aBlockIdx * aBlockRowSize * aBlockColSize;

        DT aDataElement = aBlockData[RowMjIdx(inBlockRow, inBlockCol, aBlockColSize)];
        unsigned int ar = aDenseRowBase + inBlockRow;
        unsigned int ac = aDenseColBase + inBlockCol;

        for (unsigned bc = 0; bc < bNumCols; bc++) {
            // ! This can be improved. Accumulate locally
            atomicAdd(&cData[RowMjIdx(ar, bc, bNumCols)], aDataElement * bData[RowMjIdx(ac, bc, bNumCols)]);
        }
    }
}

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmBSRWrapper1(SparseMatrixBSR<DT, MT>* a, DenseMatrix<DT, MT>* b) {
    size_t rows = a->numCols, cols = b->numCols;

    // (y, x)
    dim3 block(a->blockColSize, a->blockRowSize);
    dim3 grid(a->numBlockRows);

    assert(a->onDevice && b->onDevice);

    DenseMatrix<DT, MT>* c = new DenseMatrix<DT, MT>(a->numRows, b->numCols, true);

    spmmBSRK1<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->blockRowSize, a->blockColSize,
        a->numBlockRows, a->blockRowPtrs, a->blockColIdxs,
        a->data, b->numRows, b->numCols, b->data, 
        c->data
    );

    cudaDeviceSynchronize();

    return c;
}

// instantiations
template DenseMatrix<float, uint32_t>* spmmBSRWrapper1<float, uint32_t, double>(SparseMatrixBSR<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b);


} // namespace cuspmm
