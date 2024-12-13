#include "cuda_utils.hpp"
#include "commons.hpp"
#include "formats/sparse_coo.hpp"
#include <cstdint>

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
__global__ void spmmCOOK1(MT aNumRows, MT aNumCols, MT aNumNonZero,
                                    MT *rowIdxs, MT *colIdxs, DT* aData, 
                                    MT bNumRows, MT bNumCols, DT* bData,
                                    DT* cData) {
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

template <typename DT, typename MT, typename AccT>
__global__ void spmmCOOK2(MT aNumRows, MT aNumCols, MT aNumNonZero,
                                    MT *rowIdxs, MT *colIdxs, DT* aData, 
                                    MT bNumRows, MT bNumCols, DT* bData,
                                    DT* cData) {
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

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCOOWrapper1(SparseMatrixCOO<DT, MT>* a, DenseMatrix<DT, MT>* b) {
    const size_t numNonZero = a->numNonZero;

    const size_t BLOCKSIZE = 1024;

    dim3 block(BLOCKSIZE);
    dim3 grid((numNonZero + BLOCKSIZE - 1) / BLOCKSIZE);

    assert(a->onDevice && b->onDevice);

    DenseMatrix<DT, MT>* c = new DenseMatrix<DT, MT>(a->numRows, b->numCols, true);

    spmmCOOK1<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->rowIdxs, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );
    cudaDeviceSynchronize();

    return c;
}

// Instantiation
template DenseMatrix<float, uint32_t>* spmmCOOWrapper1<float, uint32_t, double>(SparseMatrixCOO<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b);

} // namespace cuspmm
