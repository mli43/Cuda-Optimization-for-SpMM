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
DenseMatrix<DT, MT>* spmmELLWrapper1(SparseMatrixELL<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* c) {
    const size_t numValues = a->numRows * a->maxRowNnz;

    if (b->ordering == ORDERING::COL_MAJOR) {
        b->toOrdering(ORDERING::ROW_MAJOR);
    }

    const size_t BLOCKSIZE = 1024;

    dim3 block(BLOCKSIZE);
    dim3 grid((numValues + BLOCKSIZE - 1) / BLOCKSIZE);

    assert(a->onDevice && b->onDevice);

    auto t1 = std::chrono::high_resolution_clock::now();
    spmmELLK1<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->maxRowNnz, a->colIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );
    cudaDeviceSynchronize();

    auto t2 = std::chrono::high_resolution_clock::now();
    printf("%s with shape block(z=%d,y=%d,x=%d) grid(z=%d,y=%d,x=%d): %ld ns\n", __func__,
            block.z, block.y, block.x, grid.z, grid.y, grid.x, std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());

    return c;
}

template DenseMatrix<float, uint32_t>* spmmELLWrapper1<float, uint32_t, double>(SparseMatrixELL<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c);

} // namespace cuspmm
