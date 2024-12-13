#include "formats/sparse_ell.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmELLCpu(SparseMatrixELL<DT, MT>* ma, DenseMatrix<DT, MT>* mb) {
    using mt = MT;

    DenseMatrix<DT, MT>* mc = new DenseMatrix<DT, MT>(ma->numRows, mb->numCols, false);

    if (ma->onDevice || mb->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    for (mt row = 0; row < ma->numRows; row++) {
        for (mt colind = 0; colind < ma->maxRowNnz; colind++) {
            int col = ma->colIdxs[row * ma->maxRowNnz + colind];
            DT value = ma->data[row * ma->maxRowNnz + colind];

            if (col >= 0) {
            //printf("row %d, col %d, value %f\n", row, col, value);
            for (mt j = 0; j < mb->numCols; j++){
                mc->data[row * mc->numCols + j] += value * mb->data[col * mb->numCols + j];
            }
            }
        }

    }
    /*
    for (mt idx = 0; idx < ma->numNonZero; idx++) {
        mt c = ma->colIdxs[idx];
        DT value = ma->data[idx];

        for (mt j = 0; j < mb->numCols; j++) {
            mc->data[r * mc->numCols + j] += value * mb->data[c * mb->numCols + j];
        }
    }
    */

    return mc;
};

template DenseMatrix<float, uint32_t>* spmmELLCpu<float, uint32_t, double>(SparseMatrixELL<float, uint32_t>* ma, DenseMatrix<float, uint32_t>* mb);

}
