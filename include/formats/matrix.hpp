#pragma once

#include <cstdint>

#include "cuda_utils.hpp"
#include "spmm_cusparse.hpp"

namespace cuspmm {

enum ORDERING {
    ROW_MAJOR,
    COL_MAJOR,
};


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

    // FIXME: Make it pure virtual
    virtual void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) = 0;
    virtual cusparseSpMMAlg_t getCusparseAlg() = 0;
};
}