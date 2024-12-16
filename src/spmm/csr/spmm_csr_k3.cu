#include "commons.hpp"
#include "formats/matrix.hpp"
#include "formats/sparse_csr.hpp"

#define WARPSIZE 32

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
__global__ void spmmCSRK3(const MT aNumRows, const MT aNumCols,
                          const MT aNumNonZero, const MT *rowPtrs,
                          const MT *colIdxs, const DT *aData, const MT bNumRows,
                          const MT bNumCols, const DT *bData, DT *cData) {
    // ! B is row-major
    const MT cr = blockDim.y * blockIdx.y + threadIdx.y;
    const MT cc = blockDim.x * blockIdx.x + threadIdx.x;
    const MT tid = blockDim.x * threadIdx.y + threadIdx.x;
    const int laneIdx = tid & (WARPSIZE - 1);
    const int nextLaneIdx = (laneIdx + 1) % WARPSIZE;

    if (cr >= aNumRows) {
        // This line is out of bind, whole warp quits
        return ;
    }

    const MT rowBeginIdx = __ldg(rowPtrs + cr),
             rowEndIdx = __ldg(rowPtrs + cr + 1);

    AccT result = 0.f;

    const bool validCol = cc < bNumCols;

    MT offset, ac;
    DT av;
    for (int idx = rowBeginIdx; idx < rowEndIdx; idx += WARPSIZE) {
        offset = idx + laneIdx;
        if (offset < rowEndIdx) {
            // Within current row
            ac = *(colIdxs + offset);
            av = *(aData + offset);
        } else {
            ac = 0;
            av = 0;
        }
        for (int j = 0; j < WARPSIZE; ++j) {
            ac = __shfl_sync(0xFFFFFFFF, ac, nextLaneIdx);
            av = __shfl_sync(0xFFFFFFFF, av, nextLaneIdx);
            if (validCol) {
                result += av * bData[RowMjIdx(ac, cc, bNumCols)];
            }
        }
    }
    if (cc < bNumCols) {
        cData[RowMjIdx(cr, cc, bNumCols)] = result;
    }
}

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT> *spmmCSRWrapper3(SparseMatrixCSR<DT, MT> *a,
                                     DenseMatrix<DT, MT> *b,
                                     DenseMatrix<DT, MT> *ref) {
    if (b->ordering == ORDERING::COL_MAJOR) {
        b->toOrdering(ORDERING::ROW_MAJOR);
    }
    const int kernelNum = 3;
    assert(a->onDevice && b->onDevice);

    // 1. Prologue
    auto t1 = std::chrono::high_resolution_clock::now();
    size_t rows = a->numRows, cols = b->numCols;

    auto* c = new DenseMatrix<DT, MT>(rows, cols, true, ORDERING::ROW_MAJOR);

    dim3 block(WARPSIZE, 16);
    dim3 grid((cols + block.x - 1) / block.x,
                (rows + block.y - 1) / block.y);

    // 2. Kernel
    auto t2 = std::chrono::high_resolution_clock::now();
    spmmCSRK3<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->rowPtrs, a->colIdxs,
        a->data, b->numRows, b->numCols, b->data, c->data);
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

    reportTime(testcase, a->numRows, a->numCols, a->numNonZero, std::string("CSR"), 
        b->ordering, kernelNum, (double)(pro) / 1000, (double)(kernel) / 1000, (double)(epi) / 1000, correct);

    return c;
}

template DenseMatrix<float, uint32_t>* spmmCSRWrapper3<float, uint32_t, double>(SparseMatrixCSR<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c) __attribute__((used));

} // namespace cuspmm