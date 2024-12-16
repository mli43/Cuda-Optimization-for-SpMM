#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <cstdio>
#include <torch/torch.h>


#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

template <typename DT, typename DenseMatT>
inline torch::Tensor toTorch(DenseMatT* res) {
    if constexpr (std::is_same_v<DT, __half>) {
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