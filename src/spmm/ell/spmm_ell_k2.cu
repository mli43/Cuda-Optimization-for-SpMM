#include "cuda_utils.hpp"
#include "commons.hpp"
#include "formats/sparse_ell.hpp"
#include <cstdint>

#define BLOCKSIZE 1024

namespace cuspmm {

template <typename T, typename MT, typename AccT>
__global__ void spmmELLK2(MT aNumRows, MT aNumCols, MT aNumNonZero, MT aMaxColNnz,
                                    MT *rowIdxs, T* aData, 
                                    MT bNumRows, MT bNumCols, T* bData,
                                    T* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T bCol[BLOCKSIZE];

    size_t numValues = aNumCols * aMaxColNnz;

    int row, col;
    float value;
    row = -1;
    col = 0;
    value = 0;

    if (idx < numValues) {
        col = idx / aMaxColNnz;
        row = rowIdxs[idx];
        value = aData[idx];
    }

    for (int j = 0; j < bNumCols; j++) {
        for (int startRow = 0; startRow < bNumRows; startRow += blockDim.x) {
            if (startRow + threadIdx.x < bNumRows) {
                bCol[threadIdx.x] = bData[(startRow + threadIdx.x)* bNumCols + j];
            }
            else {
                bCol[threadIdx.x] = 0;
            }
            __syncthreads();

            if (idx < numValues) {
                if (row>= 0 && col - startRow >= 0 && col - startRow < blockDim.x) {
                    atomicAdd(&cData[row * bNumCols + j], value * bCol[col - startRow]);
                }
            }
            __syncthreads();
        }
    }
}

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmELLWrapper2(SparseMatrixELL<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* c) {
    const size_t numValues = a->numCols * a->maxColNnz;

    dim3 block(BLOCKSIZE);
    dim3 grid((numValues + BLOCKSIZE - 1) / BLOCKSIZE);

    assert(a->onDevice && b->onDevice);

    if (b->ordering == ORDERING::COL_MAJOR) {
        b->toOrdering(ORDERING::ROW_MAJOR);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    spmmELLK2<DT, MT, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->maxColNnz, 
        a->rowIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );
    cudaDeviceSynchronize();

    auto t2 = std::chrono::high_resolution_clock::now();
    printf("%s with shape block(z=%d,y=%d,x=%d) grid(z=%d,y=%d,x=%d): %ld ns\n", __func__,
            block.z, block.y, block.x, grid.z, grid.y, grid.x, std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());

    return c;
}

template DenseMatrix<float, uint32_t>* spmmELLWrapper2<float, uint32_t, double>(SparseMatrixELL<float, uint32_t>* a, DenseMatrix<float, uint32_t>* b, DenseMatrix<float, uint32_t>* c) __attribute__((used));
}