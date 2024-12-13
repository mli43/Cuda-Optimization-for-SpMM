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

template<typename _dataT, typename _metaT> 
class DenseMatrix : public Matrix<_dataT, _metaT> {
  public:
    using DT = _dataT;
    using MT = _metaT;
    DT *data;
    ORDERING ordering;
    DenseMatrix() : Matrix<DT, MT>() { this->data = nullptr; }

    DenseMatrix(std::string filePath);

    DenseMatrix(MT numRows, MT numCols, bool onDevice, ORDERING ordering = ORDERING::ROW_MAJOR);

    DenseMatrix(DenseMatrix<DT, MT>* source, bool onDevice);

    ~DenseMatrix();

    bool copyData(DenseMatrix<DT, MT>* source);

    void setCusparseDnMatDesc(cusparseDnMatDescr_t* matDescP);

    void assertSameShape(DenseMatrix<DT, MT>* target);

    DenseMatrix<DT, MT>* copy2Device();

    DenseMatrix<DT, MT>* copy2Host();

    bool toOrdering(ORDERING newOrdering);

    bool save2File(std::string filePath);

    bool allocateSpace(bool onDevice);

    bool freeSpace();
};

} // namespace cuspmm
