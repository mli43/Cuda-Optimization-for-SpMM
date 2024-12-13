#pragma once

#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "spmm_cusparse.hpp"

namespace cuspmm{

template <typename T> class SparseMatrixELL: public SparseMatrix<T> {
  public:
    Matrix::metadataType *colIdxs;
    Matrix::metadataType maxRowNnz;

    SparseMatrixELL();

    SparseMatrixELL(std::string colindPath, std::string valuesPath);

    SparseMatrixELL(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, Matrix::metadataType maxRowNnz, 
                    bool onDevice);

    ~SparseMatrixELL();

    void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) override;
    cusparseSpMMAlg_t getCusparseAlg() override;

    bool allocateSpace(bool onDevice);

    SparseMatrixELL<T> *copy2Device();

    DenseMatrix<T> *toDense();

};

}
