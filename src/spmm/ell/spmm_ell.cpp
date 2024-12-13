#include "formats/sparse_ell.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmELLCpu(SparseMatrixELL<DT, MT>* ma, DenseMatrix<DT, MT>* mb, DenseMatrix<DT, MT> *mc) {
    using mt = MT;

    assert(!ma->onDevice && !mb->onDevice);

    if (mb->ordering == ORDERING::COL_MAJOR) {
        mb->toOrdering(ORDERING::ROW_MAJOR);
    }

    for (mt col = 0; col < ma->numCols; col++) {
        for (mt rowind = 0; rowind < ma->maxColNnz; rowind++) {
            int row = ma->rowIdxs[col * ma->maxColNnz + rowind];
            DT value = ma->data[col * ma->maxColNnz + rowind];

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

template DenseMatrix<float, uint32_t>* spmmELLCpu<float, uint32_t, double>(SparseMatrixELL<float, uint32_t>* ma, DenseMatrix<float, uint32_t>* mb, DenseMatrix<float, uint32_t>* mc);

}
