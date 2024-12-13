#pragma once

#include "commons.hpp"
#include "engine/engine_base.hpp"
#include "formats/dense.hpp"
#include "formats/sparse_csr.hpp"
#include "engine/cusparse.hpp"

namespace cuspmm {

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCSRCpu(SparseMatrixCSR<DT, MT>* ma, DenseMatrix<DT, MT>* mb, DenseMatrix<DT, MT> *mc);

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT>* spmmCSRWrapper1(SparseMatrixCSR<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* c);

template <typename DT, typename MT, typename AccT>
DenseMatrix<DT, MT> *spmmCSRWrapper2(SparseMatrixCSR<DT, MT> *a, DenseMatrix<DT, MT> *b, DenseMatrix<DT, MT> *c);

template<typename DT, typename MT, typename AccT>
class EngineCSR : public EngineBase {
public:
    using MataT = SparseMatrixCSR<DT, MT>;
    using MatbT = DenseMatrix<DT, MT>;

    bool SUPPORT_CUSPARSE = true;
    std::string fmt;
    EngineCSR(std::string dirPath) {
        this->numKernels = 2;
        this->fmt = dirPath + "/csr";
    }

    void* runKernel(int num, void* _ma, void* _mb, void* _mc) {
        auto ma = reinterpret_cast<MataT*>(_ma);
        auto mb = reinterpret_cast<MatbT*>(_mb);
        auto mc = reinterpret_cast<MatbT*>(_mc);
        if (num == -1) {
            // Test all cuda kernels
            spmmCSRWrapper1<DT, MT, AccT>(ma, mb, mc);
            return spmmCSRWrapper2<DT, MT, AccT>(ma, mb, mc);
        } else if (num == 0) {
            return spmmCSRCpu<DT, MT, AccT>(ma, mb, mc);
        } else if (num == 1) {
            return spmmCSRWrapper1<DT, MT, AccT>(ma, mb, mc);
        }

        throw std::runtime_error("Not implemented");
    }

};

}