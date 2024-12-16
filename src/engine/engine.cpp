#include "format.hpp"
#include "formats/matrix.hpp"
#include "engine/cusparse.hpp"
#include "engine/engine_bsr.hpp"
#include "torch/torch.h"
#include "utils.hpp"
#include <ATen/ops/miopen_convolution_transpose_ops.h>
#include <cstdint>
#include <memory>
#include <string>
#include <torch/types.h>
#include <type_traits>
#include "engine.hpp"

namespace cuspmm {
template <typename EngT>
void runEngine(EngT* engine, typename EngT::MataT* a, typename EngT::MatbT* b, float abs_tol, float rel_tol, bool skipSeq) {
    using ma_t = typename EngT::MataT;
    using mb_t = typename EngT::MatbT;
    mb_t* c = new mb_t(a->numRows, b->numCols, false, ORDERING::ROW_MAJOR);

    // 1. Move to device
    ma_t* da = a->copy2Device();
    mb_t* db = b->copy2Device();
    mb_t* dc = c->copy2Device();

    // 2. Run CPU version
    auto seqStart = std::chrono::high_resolution_clock::now();
    auto cpuResCpu = c;
    if (!skipSeq) {
        cpuResCpu = reinterpret_cast<mb_t*>(engine->runKernel(0, a, b, c));
    }
    auto seqEnd = std::chrono::high_resolution_clock::now();
    auto seqTime = std::chrono::duration_cast<std::chrono::microseconds>(seqEnd - seqStart);

    reportTime(testcase, a->numRows, a->numCols, a->numNonZero, engine->fmt, 
    b->ordering, 0, 0, (double)seqTime.count() / 1000.f, 0, 1);

    // 2. Launch kernel
    int numK = engine->numKernels;
    for (int i = 1; i <= numK; i++) {
        auto kRes = reinterpret_cast<mb_t*>(engine->runKernel(i, da, db, cpuResCpu));
        // ! Don't delete kRes here! Since it uses dc's mem
    }

    // Test cusparse
    if (engine->SUPPORT_CUSPARSE) {
        long pro, kernel, epi;
        cusparseTest<typename ma_t::DT, typename ma_t::MT>(reinterpret_cast<ma_t*>(da), reinterpret_cast<mb_t*>(db), reinterpret_cast<mb_t*>(dc), pro, kernel, epi);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto* tmp = dc->copy2Host();
        auto t2 = std::chrono::high_resolution_clock::now();
        epi += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        reportTime(testcase, a->numRows, a->numCols, a->numNonZero, engine->fmt, 
        db->ordering, -1, (double)(pro) / 1000, (double)(kernel) / 1000, (double)(epi) / 1000, 1);
    }

    // Release memory
    delete c;
    delete dc;
}

#define ENG_INST(fmt, dt, mt, acct) \
template void runEngine<Engine##fmt<dt, mt, acct>>(Engine##fmt<dt, mt, acct>* engine, Engine##fmt<dt, mt, acct>::MataT* a, Engine##fmt<dt, mt, acct>::MatbT* b, float abs_tol, float rel_tol, bool skipSeq); \

// BSR
ENG_INST(BSR, float, uint32_t, double);
ENG_INST(BSR, double, uint32_t, double);

// COO
ENG_INST(COO, float, uint32_t, double);
ENG_INST(COO, double, uint32_t, double);

// CSR
ENG_INST(CSR, float, uint32_t, double);
ENG_INST(CSR, double, uint32_t, double);

// ELL
ENG_INST(ELL, float, uint32_t, double);
ENG_INST(ELL, double, uint32_t, double);

} // namespace cuspmm