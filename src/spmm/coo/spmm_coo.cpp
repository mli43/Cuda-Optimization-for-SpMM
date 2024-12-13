#include "commons.hpp"
#include "formats/sparse_coo.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCOOCpu(SparseMatrixCOO<DT, MT>* ma, DenseMatrix<DT, MT>* mb, DenseMatrix<DT, MT> *mc) {
    using mt = MT;

    assert(!ma->onDevice && !mb->onDevice);

    for (mt idx = 0; idx < ma->numNonZero; idx++) {
        mt r = ma->rowIdxs[idx];
        mt c = ma->colIdxs[idx];
        DT value = ma->data[idx];

        for (mt j = 0; j < mb->numCols; j++) {
            mc->data[r * mc->numCols + j] += value * mb->data[c * mb->numCols + j];
        }
    }

    return mc;
}

template DenseMatrix<float, uint32_t>* spmmCOOCpu<float, uint32_t, double>(SparseMatrixCOO<float, uint32_t>* ma, DenseMatrix<float, uint32_t>* mb, DenseMatrix<float, uint32_t>* mc);

}
