#pragma once

#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "spmm_cusparse.hpp"

namespace cuspmm {

template <typename T> class SparseMatrixCOO : public SparseMatrix<T> {
  public:
    Matrix::metadataType *rowIdxs;
    Matrix::metadataType *colIdxs;

    SparseMatrixCOO();

    SparseMatrixCOO(std::string filePath);

    SparseMatrixCOO(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, bool onDevice);

    ~SparseMatrixCOO();

    void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) override;
    cusparseSpMMAlg_t getCusparseAlg() override;

    SparseMatrixCOO<T> *copy2Device(); 

    bool allocateSpace(bool onDevice);

    DenseMatrix<T> *toDense();

    template<typename U>
    friend std::ostream &operator<<(std::ostream &out, SparseMatrixCOO<U> &m);
};

} // namespace cuspmm

