#pragma once

#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "spmm_cusparse.hpp"

namespace cuspmm{

template<typename _dataT, typename _metaT>    
class SparseMatrixELL: public SparseMatrix<_dataT, _metaT> {
  public:
    using DT = _dataT;
    using MT = _metaT;
    MT *colIdxs;
    MT maxRowNnz;

    SparseMatrixELL();

    SparseMatrixELL(std::string colindPath, std::string valuesPath);

    SparseMatrixELL(MT numRows, MT numCols,
                    MT numNonZero, MT maxRowNnz, 
                    bool onDevice);

    ~SparseMatrixELL();

    void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) override;
    cusparseSpMMAlg_t getCusparseAlg() override;

    bool allocateSpace(bool onDevice);

    SparseMatrixELL<DT, MT> *copy2Device();

    DenseMatrix<DT, MT> *toDense();
};

}
