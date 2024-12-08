#pragma once

#include <cstdint>

#include "cuda_utils.hpp"

namespace cuspmm {


class Matrix{
public:
    using metadataType = uint32_t;
    metadataType numRows, numCols;
    bool onDevice;
    Matrix() {
        this->numRows = 0;
        this->numCols = 0;
        this->onDevice = false;
    }
};

template<typename T>
class SparseMatrix : public Matrix{
public:
    metadataType numNonZero;
    T* data;
    SparseMatrix() : Matrix(){
        this->numNonZero = 0;
        this->data = nullptr;
    }
};
}