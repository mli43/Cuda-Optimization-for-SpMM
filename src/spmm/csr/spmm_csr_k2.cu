#include "commons.hpp"
#include "formats/matrix.hpp"
#include "formats/sparse_csr.hpp"

#define WARPSIZE 32

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
__global__ void spmmCSRK2(const MT aNumRows, const MT aNumCols,
                          const MT aNumNonZero, const MT *rowPtrs,
                          const MT *colIdxs, const DT *aData, const MT bNumRows,
                          const MT bNumCols, const DT *bData, DT *cData) {
    // ! B must be col-major
    const MT cr = blockDim.y * blockIdx.y + threadIdx.y;
    const MT cc = blockDim.x * blockIdx.x + threadIdx.x;
    const MT tid = blockDim.x * blockIdx.y + blockIdx.x;
    const int laneIdx = tid & (WARPSIZE - 1);
    const int nextLaneIdx = (laneIdx - 1) % WARPSIZE;

    if (cr >= aNumRows) {
        // This line is out of bind, whole warp quits
        return ;
    }

    const MT rowBeginIdx = __ldg(rowPtrs + cr),
             rowEndIdx = __ldg(rowPtrs + cr + 1);

    AccT result = 0.f, v = 0.f;

    const DT *myBCol = bData + cc * bNumRows;

    MT offset, ac;
    DT av;
    for (int idx = rowBeginIdx; idx < rowEndIdx; idx += WARPSIZE) {
        offset = idx + laneIdx;
        if (offset < rowEndIdx) {
            // Within current row
            ac = __ldg(colIdxs + offset);
            av = __ldg(aData + offset);
        } else {
            ac = 0;
            av = 0;
        }
#pragma unroll
        for (int j = 0; j < WARPSIZE - 1; ++j) {
            ac = __shfl_sync(0xFFFFFFFF, ac, nextLaneIdx);
            av = __shfl_sync(0xFFFFFFFF, av, nextLaneIdx);
            if (cc < bNumCols) {
                result += ac * __ldg(myBCol + ac);
            }
        }
    }
    if (cc < bNumCols) {
        cData[RowMjIdx(cr, cc, bNumCols)] = result;
    }
}

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT> *spmmCSRWrapper2(SparseMatrixCSR<DT, MT> *a,
                                     DenseMatrix<DT, MT> *b,
                                     DenseMatrix<DT, MT> *c) {
    // FIXME: BUG!
    assert(a->onDevice && b->onDevice);

    if (b->ordering == ORDERING::ROW_MAJOR) {
        b->toOrdering(ORDERING::COL_MAJOR);
    }

    size_t rows = a->numCols, cols = b->numCols;

    for (int s = 2; s <= 16; s *= 2) {
        dim3 block(WARPSIZE, s);
        dim3 grid((cols + block.x - 1) / block.x,
                  (rows + block.y - 1) / block.y);


        auto t1 = std::chrono::high_resolution_clock::now();
        spmmCSRK2<DT, MT, AccT><<<grid, block>>>(
            a->numRows, a->numCols, a->numNonZero, a->rowPtrs, a->colIdxs,
            a->data, b->numRows, b->numCols, b->data, c->data);
        cudaDeviceSynchronize();

        auto t2 = std::chrono::high_resolution_clock::now();
        printf("%s with shape block(z=%d,y=%d,x=%d) grid(z=%d,y=%d,x=%d): %ld "
               "ns\n",
               __func__, block.z, block.y, block.x, grid.z, grid.y, grid.x,
               std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                   .count());
    }

    return c;
}

template DenseMatrix<float, uint32_t>* spmmCSRWrapper2<float, uint32_t, double>(SparseMatrixCSR<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c);

} // namespace cuspmm