#pragma once

#include "cuda_runtime.h"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "commons.hpp"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cuspmm {

template <typename T> class SparseMatrixBSR : public SparseMatrix<T> {
  public:
    Matrix::metadataType blockRowSize;
    Matrix::metadataType blockColSize;
    Matrix::metadataType numBlocks;
    Matrix::metadataType *blockRowPtrs;
    Matrix::metadataType *blockColIdxs;
    // Can be calculated based on numbers above
    Matrix::metadataType numBlockRows;
    Matrix::metadataType numElements;

    SparseMatrixBSR();

    SparseMatrixBSR(std::string filePath);

    SparseMatrixBSR(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, Matrix::metadataType blockRowSize, Matrix::metadataType blockColSize,
                    Matrix::metadataType numBlocks, bool onDevice); 

    SparseMatrixBSR(SparseMatrixBSR<T>* target, bool onDevice);

    ~SparseMatrixBSR();

    bool copyData(SparseMatrixBSR<T>* source, bool onDevice);

    SparseMatrixBSR<T> *copy2Device();

    void assertCheck();

    void assertSameShape(SparseMatrixBSR<T>* target);

    bool allocateSpace(bool onDevice);

    SparseMatrixBSR<T>* fromDense(DenseMatrix<T>* dense, Matrix::metadataType blockRowSize, Matrix::metadataType blockColSize);

    DenseMatrix<T> *toDense();

    template<typename U>
    friend std::ostream &operator<<(std::ostream &out, SparseMatrixBSR<U> &m);
};

} // namespace cuspmm
