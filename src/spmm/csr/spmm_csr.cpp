#include "formats/sparse_csr.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCSRCpu(SparseMatrixCSR<DT, MT>* ma, DenseMatrix<DT, MT>* mb, DenseMatrix<DT, MT> *mc) {
    using mt = MT;

    assert(!ma->onDevice && !mb->onDevice);

    if (mb->ordering == ORDERING::COL_MAJOR) {
        mb->toOrdering(ORDERING::ROW_MAJOR);
    }

    for (mt r = 0; r < ma->numRows; r++) {
        mt row_start = ma->rowPtrs[r];
        mt row_end = ma->rowPtrs[r+1];

        for (mt c = 0; c < mb->numCols; c++) {
            AccT acc = 0.f;
            for (mt c_idx = row_start; c_idx < row_end; c_idx++) {
                mt k = ma->colIdxs[c_idx];
                acc += ma->data[c_idx] * mb->data[k * mb->numCols + c];
            }
            mc->data[r * mc->numCols + c] = acc;
        }
    }

    return mc;
}

template __attribute__((used)) DenseMatrix<float, uint32_t>* spmmCSRCpu<float, uint32_t, double>(SparseMatrixCSR<float, uint32_t>* ma, DenseMatrix<float, uint32_t>* mb, DenseMatrix<float, uint32_t>* mc);

}