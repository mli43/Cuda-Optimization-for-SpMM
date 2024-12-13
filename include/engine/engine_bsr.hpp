#pragma once

#include "commons.hpp"
#include "engine/engine_base.hpp"
#include "formats/dense.hpp"
#include "formats/sparse_bsr.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT> *spmmBSRCpu(SparseMatrixBSR<DT, MT> *ma, DenseMatrix<DT, MT> *mb, DenseMatrix<DT, MT> *mc);

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmBSRWrapper1(SparseMatrixBSR<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* c);

template<typename DT, typename MT, typename AccT>
class EngineBSR : public EngineBase {
public:
    using MataT = SparseMatrixBSR<DT, MT>;
    using MatbT = DenseMatrix<DT, MT>;

    bool SUPPORT_CUSPARSE = false;
    std::string fmt;
    EngineBSR(std::string dirPath) {
        this->numKernels = 2;
        this->fmt = dirPath + "/bsr";
    }

    void* runKernel(int num, void* _ma, void* _mb, void* _mc) {
        auto ma = reinterpret_cast<MataT*>(_ma);
        auto mb = reinterpret_cast<MatbT*>(_mb);
        auto mc = reinterpret_cast<MatbT*>(_mc);
        if (num == 0) {
            return spmmBSRCpu<DT, MT, AccT>(ma, mb, mc);
        } else if (num == 1) {
            return spmmBSRWrapper1<DT, MT, AccT>(ma, mb, mc);
        }

        throw std::runtime_error("Not implemented");
    }

};

}