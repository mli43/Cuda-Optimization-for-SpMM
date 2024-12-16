#include "cuda_utils.hpp"
#include "commons.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "formats/sparse_csr.hpp"
#include "utils.hpp"
#include <cstdint>

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
__global__ void spmmCSRK1(MT aNumRows, MT aNumCols, MT aNumNonZero,
                                    MT *rowPtrs, MT *colIdxs, DT* aData, 
                                    MT bNumRows, MT bNumCols, DT* bData,
                                    DT* cData) {
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
            DT aValue = aData[i];
            acc += aValue * bData[c_idx * bNumCols + c];
        }
        cData[r * bNumCols + c] = acc;
    }
}

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCSRWrapper1(SparseMatrixCSR<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* ref) {
    if (b->ordering == ORDERING::COL_MAJOR) {
        b->toOrdering(ORDERING::ROW_MAJOR);
    }
    const int kernelNum = 1;
    assert(a->onDevice && b->onDevice);

    // 1. Prologue
    auto t1 = std::chrono::high_resolution_clock::now();
    size_t rows = a->numRows, cols = b->numCols;
    const size_t BLOCKSIZE = 32;

    auto* c = new DenseMatrix<DT, MT>(rows, cols, true, ORDERING::ROW_MAJOR);

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((cols + BLOCKSIZE - 1) / BLOCKSIZE, (rows + BLOCKSIZE - 1) / BLOCKSIZE);

    // 2. Kernel
    auto t2 = std::chrono::high_resolution_clock::now();
    spmmCSRK1<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->rowPtrs, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
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

    reportTime(testcase, a->numRows, a->numCols, a->numNonZero, std::string("CSR"), 
        b->ordering, kernelNum, (double)(pro) / 1000, (double)(kernel) / 1000, (double)(epi) / 1000, correct);
    
    return c;
}

template DenseMatrix<float, uint32_t>* spmmCSRWrapper1<float, uint32_t, double>(SparseMatrixCSR<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c) __attribute__((used));

} // namespace cuspmm
