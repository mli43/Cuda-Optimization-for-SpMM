#pragma once

#include "commons.hpp"
#include "engine/engine_base.hpp"
#include "formats/dense.hpp"
#include "formats/sparse_bsr.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT> *spmmBSRCpu(SparseMatrixBSR<DT, MT> *ma, DenseMatrix<DT, MT> *mb);

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmBSRWrapper1(SparseMatrixBSR<DT, MT>* a, DenseMatrix<DT, MT>* b);

template<typename DT, typename MT, typename AccT>
class EngineBSR : public EngineBase {
public:
    using MataT = SparseMatrixBSR<DT, MT>;
    using MatbT = DenseMatrix<DT, MT>;
    std::string fmt;
    EngineBSR(std::string dirPath) {
        this->numKernels = 2;
        this->fmt = dirPath + "/bsr";
    }

    void* runKernel(int num, void* _ma, void* _mb) {
        auto ma = reinterpret_cast<MataT*>(_ma);
        auto mb = reinterpret_cast<MatbT*>(_mb);
        if (num == 0) {
            return spmmBSRCpu<DT, MT, AccT>(ma, mb);
        } else if (num == 1) {
            return spmmBSRWrapper1<DT, MT, AccT>(ma, mb);
        }

        throw std::runtime_error("Not implemented");
    }

};

}