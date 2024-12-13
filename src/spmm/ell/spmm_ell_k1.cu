#include "cuda_utils.hpp"
#include "commons.hpp"
#include "formats/sparse_ell.hpp"
#include <cstdint>

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
__global__ void spmmELLK1(MT aNumRows, MT aNumCols, MT aNumNonZero, MT aMaxRowNnz,
                                    MT *colIdxs, DT* aData, 
                                    MT bNumRows, MT bNumCols, DT* bData,
                                    DT* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t numValues = aNumRows * aMaxRowNnz;

    if (idx < numValues) {
        int row = idx / aMaxRowNnz;
        int col = colIdxs[idx];
        float value = aData[idx];
        

        if (col >= 0) {
            for (int j = 0; j < bNumCols; j++) {
                atomicAdd(&cData[row * bNumCols + j], value * bData[col * bNumCols + j]); 
            }
        }
    }
}

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmELLWrapper1(SparseMatrixELL<DT, MT>* a, DenseMatrix<DT, MT>* b) {
    const size_t numValues = a->numRows * a->maxRowNnz;

    const size_t BLOCKSIZE = 1024;

    dim3 block(BLOCKSIZE);
    dim3 grid((numValues + BLOCKSIZE - 1) / BLOCKSIZE);

    assert(a->onDevice && b->onDevice);
    DenseMatrix<DT, MT>* c = new DenseMatrix<DT, MT>(a->numRows, b->numCols, true);

    spmmELLK1<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->maxRowNnz, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );
    cudaDeviceSynchronize();

    return c;
}

template DenseMatrix<float, uint32_t>* spmmELLWrapper1<float, uint32_t, double>(SparseMatrixELL<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b);

} // namespace cuspmm
