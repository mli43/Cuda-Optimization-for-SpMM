#include "spmm_bsr.hpp"
#include "commons.hpp"

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmCsrCpu(SparseMatrixBSR<T>* ma, DenseMatrix<T>* mb) {
    using mt = Matrix::metadataType;

    DenseMatrix<T>* mc = new DenseMatrix<T>(ma->numRows, mb->numCols, false);

    if (ma->onDevice || mb->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    for (mt blockRow = 0; blockRow < ma->numBlockRows; blockRow++) {
        mt blockRowStart = ma->blockRowPtrs[blockRow];
        mt blockRowEnd = ma->blockRowPtrs[blockRow + 1];
        for (mt blockIdx = blockRowStart; blockIdx < blockRowEnd; blockIdx++) {
            mt blockCol = ma->blockColIdxs[blockIdx];
            T* blockData = ma->data + (ma->blockSize * ma->blockSize * blockIdx);

            const mt denseRowStart = blockRow * ma->blockSize;
            const mt denseRowEnd = denseRowStart + ma->blockSize;
            const mt denseColStart = blockCol * ma->blockSize;
            const mt denseColEnd = denseColStart + ma->blockSize;
            for (mt ar = denseRowStart; ar < denseRowEnd; ar++) {
                for (mt ac = denseColStart; ac < denseColEnd; ac++) {
                    for (mt bc = 0; bc < mb->numCols; bc++) {
                        mc->data[RrowMjIdx(ar, bc, mc->numCols)] += blockData[RrowMjIdx(ar - denseRowStart, ac - denseColStart, ma->blockSize)] * mb->data[RrowMjIdx(ac, bc, mb->numCols)];
                    }
                }
            }
        }
    }

    return mc;
}

template DenseMatrix<float>* spmmCsrCpu<float, double>(SparseMatrixBSR<float>* ma, DenseMatrix<float>* mb);

}