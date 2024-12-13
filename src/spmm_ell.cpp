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

    for (mt col = 0; col < ma->numCols; col++) {
        for (mt rowind = 0; rowind < ma->maxColNnz; rowind++) {
            int row = ma->rowIdxs[col * ma->maxColNnz + rowind];
            T value = ma->data[col * ma->maxColNnz + rowind];

            if (row>= 0) {
            //printf("row %d, col %d, value %f\n", row, col, value);
                for (mt j = 0; j < mb->numCols; j++){
                    mc->data[row * mc->numCols + j] += value * mb->data[col * mb->numCols + j];
                }
            }
        }

    }

    return mc;
};

template DenseMatrix<float>* spmmEllCpu<float, double>(SparseMatrixELL<float>* ma, DenseMatrix<float>* mb);

}
