#pragma once

#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "spmm_cusparse.hpp"

namespace cuspmm {

template<typename _dataT, typename _metaT>   
class SparseMatrixCSR : public SparseMatrix<_dataT, _metaT> {
  public:
    using DT = _dataT;
    using MT = _metaT;
    MT *rowPtrs;
    MT *colIdxs;

    SparseMatrixCSR();

    SparseMatrixCSR(std::string filePath);

    SparseMatrixCSR(MT numRows, MT numCols,
                    MT numNonZero, bool onDevice);

    ~SparseMatrixCSR();

    void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) override;
    cusparseSpMMAlg_t getCusparseAlg() override;

    SparseMatrixCSR<DT, MT> *copy2Device();

    bool allocateSpace(bool onDevice);

    DenseMatrix<DT, MT> *toDense();

    template<typename U, typename MT>
    friend std::ostream &operator<<(std::ostream &out, SparseMatrixCSR<U, MT> &m);
};

} // namespace cuspmm
