#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/matrix.hpp"
#include "formats/sparse_csr.hpp"
#include <cstddef>

#define WARPSIZE 32
#define MAXSIZE 4096

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
__global__ void spmmCSRK4(const MT aNumRows, const MT aNumCols,
                          const MT aNumNonZero, const MT *rowPtrs,
                          const MT *colIdxs, const DT *aData, const MT bNumRows,
                          const MT bNumCols, const DT *bData, DT *cData) {
    // ! B is col major
    const MT cr = blockDim.y * blockIdx.y + threadIdx.y;
    const MT cStartCol = blockDim.x * blockIdx.x;
    const MT cEndCol = blockDim.x * (blockIdx.x + 1) < bNumCols ? blockDim.x * (blockIdx.x + 1) : bNumCols;
    const MT tid = blockDim.x * threadIdx.y + threadIdx.x;
    const int laneIdx = tid & (WARPSIZE - 1);
    const int nextLaneIdx = (laneIdx + 1) % WARPSIZE;
    const int threadNum = blockDim.x * blockDim.y;

    __shared__ DT bCol_s[MAXSIZE];
    __shared__ MT aColIdxs_s[MAXSIZE / 2];
    __shared__ DT aVal_s[MAXSIZE / 2];

    AccT result = 0.f;
    MT offset, ac;
    DT av;

    MT rowBeginIdx = 0, rowEndIdx = 0;
    if (cr < aNumRows) {
        rowBeginIdx = __ldg(rowPtrs + cr);
        rowEndIdx = __ldg(rowPtrs + cr + 1);
    }

    unsigned rowNumNonZero = rowEndIdx - rowBeginIdx;

    // Load A data;
    MT* myAColIdxs_s = &aColIdxs_s[RowMjIdx(threadIdx.y, 0, MAXSIZE / 2 / blockDim.y)];
    DT* myAVal_s = &aVal_s[RowMjIdx(threadIdx.y, 0, MAXSIZE / 2 / blockDim.y)];
    if (cr < aNumRows) {
        // Load a colidx and data into shared memory
        for (int idx = laneIdx; idx < rowNumNonZero; idx += WARPSIZE) {
            offset = rowBeginIdx + idx;
            myAColIdxs_s[idx] = colIdxs[offset];
            myAVal_s[idx] = aData[offset];
        }
    }
    __syncthreads();

    for (int cc = cStartCol; cc < cEndCol; cc++) {
        // Warp load b col
        int gidx = tid;
        while (gidx < bNumRows) {
            bCol_s[gidx] = bData[ColMjIdx(gidx, cc, bNumRows)];
            gidx += threadNum;
        }
        __syncthreads();

        result = 0.f;
        if (cr < aNumRows) {
            for (int idx = laneIdx; idx < rowNumNonZero; idx += WARPSIZE) {
                result += myAVal_s[idx] * bCol_s[myAColIdxs_s[idx]];
            }
            
            // reduce
            for (int gap = 16; gap > 0; gap >>= 1) {
                result += __shfl_sync(0xFFFFFFFF, result, (laneIdx + gap) % WARPSIZE);
            }
            if (laneIdx == 0) {
                cData[RowMjIdx(cr, cc, bNumCols)] = result;
            }
        }
    }
}

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT> *spmmCSRWrapper4(SparseMatrixCSR<DT, MT> *a,
                                     DenseMatrix<DT, MT> *b,
                                     DenseMatrix<DT, MT> *ref) {
    if (b->ordering == ORDERING::ROW_MAJOR) {
        b->toOrdering(ORDERING::COL_MAJOR);
    }
    const int kernelNum = 4;
    assert(a->onDevice && b->onDevice);

    int rowSize = 2;
    double sparsity = (double)(a->numNonZero) / (a->numRows * a->numCols);
    const double rate = 2;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = t1, t3 = t1;
    if (sparsity * a->numCols * rate > (double)MAXSIZE / 2 / rowSize || b->numRows > MAXSIZE) {
        reportTime(testcase, a->numRows, a->numCols, a->numNonZero, std::string("CSR"), 
            b->ordering, kernelNum, 0, 0, 0, 0);
        return nullptr;
    }

    // 1. Prologue
    size_t rows = a->numRows, cols = b->numCols;

    auto* c = new DenseMatrix<DT, MT>(rows, cols, true, ORDERING::ROW_MAJOR);

    dim3 block(WARPSIZE, rowSize);
    dim3 grid((cols + block.x - 1) / block.x,
                (rows + block.y - 1) / block.y);

    // 2. Kernel
    t2 = std::chrono::high_resolution_clock::now();
    spmmCSRK4<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->rowPtrs, a->colIdxs,
        a->data, b->numRows, b->numCols, b->data, c->data);
    cudaCheckError(cudaDeviceSynchronize());
    // 3. Epilogue
    t3 = std::chrono::high_resolution_clock::now();
    // printf("%s with shape block(z=%d,y=%d,x=%d) grid(z=%d,y=%d,x=%d): %ld ns\n", __func__,
    //         block.z, block.y, block.x, grid.z, grid.y, grid.x, std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());
    // auto copy = c->copy2Host();
    // copy->save2File("CSR4.res");
    // delete copy;
    
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

template DenseMatrix<float, uint32_t>* spmmCSRWrapper4<float, uint32_t, double>(SparseMatrixCSR<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c) __attribute__((used));

} // namespace cuspmm