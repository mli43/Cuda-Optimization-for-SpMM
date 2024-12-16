#pragma once

#include "commons.hpp"
#include "engine/engine_base.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "formats/sparse_bsr.hpp"
#include <cstdio>

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
    std::string dirPath;
    double seqTime = 1.f;
    EngineBSR(std::string dirPath) {
        this->numKernels = 1;
        this->dirPath = dirPath;
        this->fmt = "BSR";
    }
    void logSeq(double seq) {
        this->seqTime = seq;
    }

    void report(MataT* a, MatbT* b, int num, double pro, double kernel, double epilog, bool correct) {
        std::string ord;
        if (b->ordering == ORDERING::ROW_MAJOR) {
            ord = "ROW_MAJOR";
        } else {
            ord = "COL_MAJOR";
        }
        double total = pro + kernel + epilog;
        std::cout << "{\n\"testcase\":\"" << this->dirPath << "\",\n" 
                    << "\"sparsity\":\"" << ((double)a->numNonZero / (a->numRows * a->numCols)) << "\",\n"
                    << "\"format\":\"" << this->fmt << "\",\n"
                    << "\"kernelType\":\"" << num << "\",\n"
                    << "\"denseOrdering\":\"" << ord << "\",\n"
                    << "\"correct\":\"" << correct << "\",\n";
        printf("\"cudaPrologTimeMs\":\"%lf\",\n"
                "\"cudaKernelTimeMs\":\"%lf\",\n"
                "\"cudaEpilogTimeMs\":\"%lf\",\n"
                "\"cudaTotalTimeMs\":\"%lf\",\n"
                "\"sequentialTimeMs\":\"%lf\"\n},\n", pro, kernel, epilog, total, this->seqTime);
    }

    void* runKernel(int num, void* _ma, void* _mb, void* _mc) {
        auto ma = reinterpret_cast<MataT*>(_ma);
        auto mb = reinterpret_cast<MatbT*>(_mb);
        auto mc = reinterpret_cast<MatbT*>(_mc);
        if (num == -1) {
            // Test all cuda kernels
            return spmmBSRWrapper1<DT, MT, AccT>(ma, mb, mc);
        } else if (num == 0) {
            return spmmBSRCpu<DT, MT, AccT>(ma, mb, mc);
        } else if (num == 1) {
            return spmmBSRWrapper1<DT, MT, AccT>(ma, mb, mc);
        }

        throw std::runtime_error("Not implemented");
    }

};

}