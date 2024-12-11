#include "spmm_ell.hpp"
#include <cstddef>

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmEllCpu(SparseMatrixELL<T>* ma, DenseMatrix<T>* mb) {
    using mt = Matrix::metadataType;

    DenseMatrix<T>* mc = new DenseMatrix<T>(ma->numRows, mb->numCols, false);

    if (ma->onDevice || mb->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    for (mt row = 0; row < ma->numRows; row++) {
        for (mt colind = 0; colind < ma->maxNumNnz; colind++) {
            mt col = ma->colIdxs[row * ma->maxNumNnz + colind];
            T value = ma->data[row * ma->maxNumNnz + colind];

            for (mt j = 0; j < mb->numCols; j++){
                mc->data[row * mc->numCols + j] += value * mb->data[col * mb->numCols + j];
            }
        }

    }
    /*
    for (mt idx = 0; idx < ma->numNonZero; idx++) {
        mt c = ma->colIdxs[idx];
        T value = ma->data[idx];

        for (mt j = 0; j < mb->numCols; j++) {
            mc->data[r * mc->numCols + j] += value * mb->data[c * mb->numCols + j];
        }
    }
    */

    return mc;
}

template DenseMatrix<float>* spmmEllCpu<float, double>(SparseMatrixCOO<float>* ma, DenseMatrix<float>* mb);

}
