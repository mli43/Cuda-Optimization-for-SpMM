#pragma once

#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "spmm_cusparse.hpp"

namespace cuspmm {

template <typename T> class SparseMatrixCSR : public SparseMatrix<T> {
  public:
    Matrix::metadataType *rowPtrs;
    Matrix::metadataType *colIdxs;

    SparseMatrixCSR();

    SparseMatrixCSR(std::string filePath);

    SparseMatrixCSR(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, bool onDevice);

    ~SparseMatrixCSR();

    void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) override;
    cusparseSpMMAlg_t getCusparseAlg() override;

    SparseMatrixCSR<T> *copy2Device();

    bool allocateSpace(bool onDevice);

    DenseMatrix<T> *toDense();

    template<typename U>
    friend std::ostream &operator<<(std::ostream &out, SparseMatrixCSR<U> &m);
};

} // namespace cuspmm
