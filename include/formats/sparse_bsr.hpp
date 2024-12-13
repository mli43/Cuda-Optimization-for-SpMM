#pragma once

#include "commons.hpp"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "spmm_cusparse.hpp"


namespace cuspmm {

template<typename _dataT, typename _metaT> 
class SparseMatrixBSR : public SparseMatrix<_dataT, _metaT> {
  public:
    using DT = _dataT;
    using MT = _metaT;
    MT blockRowSize;
    MT blockColSize;
    MT numBlocks;
    MT *blockRowPtrs;
    MT *blockColIdxs;
    // Can be calculated based on numbers above
    MT numBlockRows;
    MT numElements;

    SparseMatrixBSR();

    SparseMatrixBSR(std::string filePath);

    SparseMatrixBSR(MT numRows, MT numCols,
                    MT numNonZero, MT blockRowSize, MT blockColSize,
                    MT numBlocks, bool onDevice); 

    SparseMatrixBSR(SparseMatrixBSR<DT, MT>* target, bool onDevice);

    ~SparseMatrixBSR();

    void setCusparseSpMatDesc(cusparseSpMatDescr_t* matDescP) override;
    cusparseSpMMAlg_t getCusparseAlg() override;

    bool copyData(SparseMatrixBSR<DT, MT>* source, bool onDevice);

    SparseMatrixBSR<DT, MT> *copy2Device();

    void assertCheck();

    void assertSameShape(SparseMatrixBSR<DT, MT>* target);

    bool allocateSpace(bool onDevice);

    SparseMatrixBSR<DT, MT>* fromDense(DenseMatrix<DT, MT>* dense, MT blockRowSize, MT blockColSize);

    DenseMatrix<DT, MT> *toDense();

    template<typename U, typename MT>
    friend std::ostream &operator<<(std::ostream &out, SparseMatrixBSR<U, MT> &m);
};

} // namespace cuspmm
