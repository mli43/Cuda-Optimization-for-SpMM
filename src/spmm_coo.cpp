#include "spmm_coo.hpp"
#include <cstddef>

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmCooCpu(SparseMatrixCOO<T>* ma, DenseMatrix<T>* mb) {
    using mt = Matrix::metadataType;

    DenseMatrix<T>* mc = new DenseMatrix<T>(ma->numRows, mb->numCols, false);

    if (ma->onDevice || mb->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    for (mt idx = 0; idx < ma->numNonZero; idx++) {
        mt r = ma->rowIdxs[idx];
        mt c = ma->colIdxs[idx];
        T value = ma->data[idx];

        for (mt j = 0; j < mb->numCols; j++) {
            mc->data[r * mc->numCols + j] += value * mb->data[c * mb->numCols + j];
        }
    }

    return mc;
}

template DenseMatrix<float>* spmmCooCpu<float, double>(SparseMatrixCOO<float>* ma, DenseMatrix<float>* mb);

}
