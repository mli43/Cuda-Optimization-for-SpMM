#include "spmm_bsr.hpp"
#include "commons.hpp"

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T> *spmmBsrCpu(SparseMatrixBSR<T> *ma, DenseMatrix<T> *mb) {
    using mt = Matrix::metadataType;

    DenseMatrix<T> *mc = new DenseMatrix<T>(ma->numRows, mb->numCols, false);

    if (ma->onDevice || mb->onDevice) {
        std::cerr << "Device incorrect!" << std::endl;
        return nullptr;
    }

    for (mt blockRow = 0; blockRow < ma->numBlockRows; blockRow++) {
        mt blockRowStart = ma->blockRowPtrs[blockRow];
        mt blockRowEnd = ma->blockRowPtrs[blockRow + 1];
        for (mt blockIdx = blockRowStart; blockIdx < blockRowEnd; blockIdx++) {
            mt blockCol = ma->blockColIdxs[blockIdx];
            T *blockData =
                ma->data + (ma->blockRowSize * ma->blockColSize * blockIdx);

            const mt denseRowStart = blockRow * ma->blockRowSize;
            const mt denseRowEnd = denseRowStart + ma->blockRowSize;
            const mt denseColStart = blockCol * ma->blockColSize;
            const mt denseColEnd = denseColStart + ma->blockColSize;
            for (mt ar = denseRowStart; ar < denseRowEnd; ar++) {
                for (mt ac = denseColStart; ac < denseColEnd; ac++) {
                    for (mt bc = 0; bc < mb->numCols; bc++) {
                        mc->data[RowMjIdx(ar, bc, mc->numCols)] +=
                            blockData[RowMjIdx(ar - denseRowStart, ac - denseColStart, ma->blockColSize)] *
                            mb->data[RowMjIdx(ac, bc, mb->numCols)];
                    }
                }
            }
        }
    }

    return mc;
}

template DenseMatrix<float> *
spmmBsrCpu<float, double>(SparseMatrixBSR<float> *ma, DenseMatrix<float> *mb);

} // namespace cuspmm