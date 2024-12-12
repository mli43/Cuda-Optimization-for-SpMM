#pragma once

#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/matrix.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstring>

namespace cuspmm {

template <typename T> class DenseMatrix : public Matrix {
  public:
    T *data;
    ORDERING ordering;
    DenseMatrix() : Matrix() { this->data = nullptr; }

    DenseMatrix(std::string filePath);

    DenseMatrix(Matrix::metadataType numRows,
                       Matrix::metadataType numCols, bool onDevice, ORDERING ordering = ORDERING::ROW_MAJOR);

    DenseMatrix(DenseMatrix<T>* source, bool onDevice);

    ~DenseMatrix();

    /**
     * @brief Copy only pointer-like data
     */
    bool copyData(DenseMatrix<T>* source);

    void assertSameShape(DenseMatrix<T>* target);

    DenseMatrix<T>* copy2Device();

    DenseMatrix<T>* copy2Host();

    bool toOrdering(ORDERING newOrdering);

    bool save2File(std::string filePath);

    bool allocateSpace(bool onDevice);

    bool freeSpace();
};

} // namespace cuspmm
