#include "cuda_utils.hpp"
#include "commons.hpp"
#include "formats/sparse_csr.hpp"
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
DenseMatrix<DT, MT>* spmmCSRWrapper1(SparseMatrixCSR<DT, MT>* a, DenseMatrix<DT, MT>* b) {
    size_t rows = a->numCols, cols = b->numCols;

    const size_t BLOCKSIZE = 32;

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((cols + BLOCKSIZE - 1) / BLOCKSIZE, (rows + BLOCKSIZE - 1) / BLOCKSIZE);

    assert(a->onDevice && b->onDevice);

    DenseMatrix<DT, MT>* c = new DenseMatrix<DT, MT>(a->numRows, b->numCols, true);

    spmmCSRK1<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->rowPtrs, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );
    cudaDeviceSynchronize();

    return c;
}

template DenseMatrix<float, uint32_t>* spmmCSRWrapper1<float, uint32_t, double>(SparseMatrixCSR<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b);

} // namespace cuspmm
