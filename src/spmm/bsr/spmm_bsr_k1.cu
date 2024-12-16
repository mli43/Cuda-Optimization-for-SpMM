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
DenseMatrix<DT, MT>* spmmBSRWrapper1(SparseMatrixBSR<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* ref) {
    assert(a->onDevice && b->onDevice);
    // (x, y)
    if (b->ordering == ORDERING::COL_MAJOR) {
        b->toOrdering(ORDERING::ROW_MAJOR);
    }
    const int kernelNum = 1;

    // 1. Prologue
    auto t1 = std::chrono::high_resolution_clock::now();
    size_t rows = a->numRows, cols = b->numCols;

    auto* c = new DenseMatrix<DT, MT>(rows, cols, true, ORDERING::ROW_MAJOR);

    dim3 block(a->blockRowSize, a->blockColSize);
    dim3 grid(a->numBlockRows);

    auto t2 = std::chrono::high_resolution_clock::now();
    spmmBSRK1<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->blockRowSize, a->blockColSize,
        a->numBlockRows, a->blockRowPtrs, a->blockColIdxs,
        a->data, b->numRows, b->numCols, b->data, 
        c->data
    );

    cudaDeviceSynchronize();
    // 3. Epilogue
    auto t3 = std::chrono::high_resolution_clock::now();
    // printf("%s with shape block(z=%d,y=%d,x=%d) grid(z=%d,y=%d,x=%d): %ld ns\n", __func__,
    //         block.z, block.y, block.x, grid.z, grid.y, grid.x, std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    
    auto res = c->copy2Host();
    auto t4 = std::chrono::high_resolution_clock::now();

    auto pro = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto kernel = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    auto epi = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // Check correctness
    torch::Tensor refTorch = toTorch<DT, DenseMatrix<DT, MT>>(ref);
    torch::Tensor cTorch = toTorch<DT, DenseMatrix<DT, MT>>(res);
    bool correct = torch::allclose(cTorch, refTorch, REL_TOL, ABS_TOL);

    reportTime(testcase, a->numRows, a->numCols, a->numNonZero, std::string("BSR"), 
        b->ordering, kernelNum, (double)(pro) / 1000, (double)(kernel) / 1000, (double)(epi) / 1000, correct);
    
    return c;
}

// instantiations
template DenseMatrix<float, uint32_t>* spmmBSRWrapper1<float, uint32_t, double>(SparseMatrixBSR<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c) __attribute__((used));


} // namespace cuspmm
