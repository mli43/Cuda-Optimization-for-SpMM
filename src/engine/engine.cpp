#include "engine/engine_bsr.hpp"
#include "format.hpp"
#include "torch/torch.h"
#include <cstdint>
#include <memory>
#include <torch/types.h>
#include <type_traits>
#include "engine.hpp"

namespace cuspmm {
template <typename DT, typename DenseMatT>
inline torch::Tensor toTorch(DenseMatT* res) {
    if constexpr (std::is_same_v<DT, half>) {
        auto options = torch::TensorOptions().dtype(torch::kFloat16).requires_grad(false);
        return torch::from_blob(res->data, {res->numRows, res->numCols}, options).clone();
    }
    if constexpr (std::is_same_v<DT, float>) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
        return torch::from_blob(res->data, {res->numRows, res->numCols}, options).clone();
    }
    if constexpr (std::is_same_v<DT, double>) {
        auto options = torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false);
        return torch::from_blob(res->data, {res->numRows, res->numCols}, options).clone();
    }
}

template <typename EngT>
void runEngine(EngT* engine, typename EngT::MataT* a, typename EngT::MatbT* b, float abs_tol, float rel_tol) {
    using ma_t = typename EngT::MataT;
    using mb_t = typename EngT::MatbT;
    auto t1 = std::chrono::high_resolution_clock::now();

    // 1. Move to device
    ma_t* da = a->copy2Device();
    mb_t* db = b->copy2Device();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto copy2DeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "copy2DeviceTime (us):" << copy2DeviceTime.count() << "\n";

    // 2. Run CPU version
    auto t3 = std::chrono::high_resolution_clock::now();
    auto cpuResCpu = reinterpret_cast<mb_t*>(engine->runKernel(0, a, b));
    auto t4 = std::chrono::high_resolution_clock::now();
    auto seqTime = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
    std::cout << "seqTime (us):" << seqTime.count() << "\n";

    // Create a torch version cpu result
    torch::Tensor cpuResTorch = toTorch<typename mb_t::DT, mb_t>(cpuResCpu);
    cpuResCpu->save2File(engine->fmt + "_cpu.res");

    // 2. Launch kernel
    int numKernels = engine->numKernels;
    for (int i = 1; i < numKernels; i++) {
        // 2.1 timing kernel
        auto kernel_start = std::chrono::high_resolution_clock::now();
        // FIXME: Involving allocation time now!
        auto kRes = reinterpret_cast<mb_t*>(engine->runKernel(i, da, db));
        auto kernel_end = std::chrono::high_resolution_clock::now();
        auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

        auto kResCpu = kRes->copy2Host();
        torch::Tensor kResTorch = toTorch<typename mb_t::DT, mb_t>(kResCpu);

        // 2.2 check correctness
        std::cout << "kernel " << i << " takse " << kernelTime.count() << "(us), allclose result: " <<
            torch::allclose(cpuResTorch, kResTorch, rel_tol, abs_tol) << std::endl;
        
        delete kResCpu;
    }

    // FIXME: release memory
    delete da;
    delete db;
}

#define ENG_INST(fmt, dt, mt, acct) \
template void runEngine<Engine##fmt<dt, mt, acct>>(Engine##fmt<dt, mt, acct>* engine, Engine##fmt<dt, mt, acct>::MataT* a, Engine##fmt<dt, mt, acct>::MatbT* b, float abs_tol, float rel_tol); \

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