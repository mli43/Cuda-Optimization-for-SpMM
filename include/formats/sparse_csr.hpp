#pragma once

#include "cuda_runtime.h"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
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

template <typename T> class SparseMatrixCSR : public SparseMatrix<T> {
  public:
    Matrix::metadataType *rowPtrs;
    Matrix::metadataType *colIdxs;

    SparseMatrixCSR();

    SparseMatrixCSR(std::string filePath);

    SparseMatrixCSR(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, bool onDevice);

    ~SparseMatrixCSR();

    SparseMatrixCSR<T> *copy2Device();

    bool allocateSpace(bool onDevice);

    DenseMatrix<T> *toDense();

    template<typename U>
    friend std::ostream &operator<<(std::ostream &out, SparseMatrixCSR<U> &m);
};

} // namespace cuspmm
