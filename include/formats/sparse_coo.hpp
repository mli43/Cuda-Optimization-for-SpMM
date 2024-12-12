#pragma once

#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

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

    SparseMatrixCOO<T> *copy2Device(); 

    bool allocateSpace(bool onDevice);

    DenseMatrix<T> *toDense();

    template<typename U>
    friend std::ostream &operator<<(std::ostream &out, SparseMatrixCOO<U> &m);
};

} // namespace cuspmm

