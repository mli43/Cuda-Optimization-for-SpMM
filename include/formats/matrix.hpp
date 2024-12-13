#pragma once

#include <cstdint>

#include "cuda_utils.hpp"
#include "spmm_cusparse.hpp"

namespace cuspmm {

enum ORDERING {
    ROW_MAJOR,
    COL_MAJOR,
};


template<typename _dataT, typename _metaT>
class Matrix{
public:
    using DT = _dataT;
    using MT = _metaT;
    MT numRows, numCols;
    bool onDevice;

    Matrix() {
        this->numRows = 0;
        this->numCols = 0;
        this->onDevice = false;
    }

};

template<typename _dataT, typename _metaT>
class SparseMatrix : public Matrix<_dataT, _metaT>{
public:
    using DT = _dataT;
    using MT = _metaT;
    MT numNonZero;
    DT* data;
    SparseMatrix() : Matrix<_dataT, _metaT>(){
        this->numNonZero = 0;
        this->data = nullptr;
    }

    // FIXME: Make it pure virtual
    virtual void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) = 0;
    virtual cusparseSpMMAlg_t getCusparseAlg() = 0;
};
}