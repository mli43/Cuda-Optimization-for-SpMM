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

    bool allocateSpace(bool onDevice);

    SparseMatrixELL<T> *copy2Device();

    DenseMatrix<T> *toDense();

};

}
