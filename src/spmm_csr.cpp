#include "spmm_csr.hpp"

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmCsrCpu(SparseMatrixCSR<T>* ma, DenseMatrix<T>* mb) {
    using mt = Matrix::metadataType;

    DenseMatrix<T>* mc = new DenseMatrix<T>(ma->numRows, mb->numCols, false);

    if (ma->onDevice || mb->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
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

template DenseMatrix<float>* spmmCsrCpu<float, double>(SparseMatrixCSR<float>* ma, DenseMatrix<float>* mb);

}