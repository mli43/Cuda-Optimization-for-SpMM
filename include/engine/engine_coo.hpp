#pragma once

#include "commons.hpp"
#include "engine/engine_base.hpp"
#include "formats/sparse_coo.hpp"
#include "engine/cusparse.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCOOCpu(SparseMatrixCOO<DT, MT>* ma, DenseMatrix<DT, MT>* mb, DenseMatrix<DT, MT> *mc);

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCOOWrapper1(SparseMatrixCOO<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* c);

template<typename DT, typename MT, typename AccT>
class EngineCOO : public EngineBase {
public:
    using MataT = SparseMatrixCOO<DT, MT>;
    using MatbT = DenseMatrix<DT, MT>;

    bool SUPPORT_CUSPARSE = true;
    std::string fmt;
    EngineCOO(std::string dirPath) {
        this->numKernels = 2;
        this->fmt = dirPath + "/coo";
    }

    void* runKernel(int num, void* _ma, void* _mb, void* _mc) {
        auto ma = reinterpret_cast<MataT*>(_ma);
        auto mb = reinterpret_cast<MatbT*>(_mb);
        auto mc = reinterpret_cast<MatbT*>(_mc);
        if (num == -1) {
            // Test all cuda kernels
            return spmmCOOWrapper1<DT, MT, AccT>(ma, mb, mc);
        } else if (num == 0) {
            return spmmCOOCpu<DT, MT, AccT>(ma, mb, mc);
        } else if (num == 1) {
            return spmmCOOWrapper1<DT, MT, AccT>(ma, mb, mc);
        }

        throw std::runtime_error("Not implemented");
    }

};

}