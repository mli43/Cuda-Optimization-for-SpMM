#pragma once

#include "commons.hpp"
#include "engine/engine_base.hpp"
#include "formats/sparse_coo.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCOOCpu(SparseMatrixCOO<DT, MT>* ma, DenseMatrix<DT, MT>* mb);

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCOOWrapper1(SparseMatrixCOO<DT, MT>* a, DenseMatrix<DT, MT>* b);

template<typename DT, typename MT, typename AccT>
class EngineCOO : public EngineBase {
public:
    using MataT = SparseMatrixCOO<DT, MT>;
    using MatbT = DenseMatrix<DT, MT>;
    std::string fmt;
    EngineCOO(std::string dirPath) {
        this->numKernels = 2;
        this->fmt = dirPath + "/coo";
    }

    void* runKernel(int num, void* _ma, void* _mb) {
        auto ma = reinterpret_cast<MataT*>(_ma);
        auto mb = reinterpret_cast<MatbT*>(_mb);
        if (num == 0) {
            return spmmCOOCpu<DT, MT, AccT>(ma, mb);
        } else if (num == 1) {
            return spmmCOOWrapper1<DT, MT, AccT>(ma, mb);
        }

        throw std::runtime_error("Not implemented");
    }

};

}