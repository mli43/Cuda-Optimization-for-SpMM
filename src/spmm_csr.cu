#include "cuda_utils.hpp"
#include "torch/torch.h"
#include "format.hpp"
#include "loader.hpp"
#include "cuda.h"
#include "cuda_runtime.h"


namespace cuspmm {


template <typename T>
void runEngineCSR(SparseMatrixCSR<T>* hm, float abs_tol, double rel_tol) {

    SparseMatrixCSR<T>* dm; 
    cudaCheckError(cudaMalloc(&dm, sizeof(SparseMatrixCSR<T>)));

    // 1. Move to device
    move2DeviceCSR(hm, dm);

    std::cout<<"Starting kernel"<<std::endl;
    spmm_csr_dsd<T>(A_d, B_d, C_d, N);
    std::cout<<"Kernel done"<<std::endl;

    CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, R * N * sizeof(T), cudaMemcpyDeviceToHost));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor A_t = torch::from_blob(sparse_matrix.mat, {R, C}, options).clone().cuda();
    torch::Tensor B_t = torch::from_blob(B_h, {C, N}, options).clone().cuda();
    torch::Tensor C_cuda = torch::from_blob(C_h, {R, N}, options).clone();

    torch::Tensor C_t = compute_torch_mm<torch::Tensor>(A_t, B_t).cpu();

    std::cout << "CUDA vs Torch allclose: "
              << (torch::allclose(C_cuda, C_t, abs_tol, rel_tol) ? "true" : "false")
              << std::endl;

    CHECK_CUDA_ERROR(cudaFree(A_d.rowPtrs));
    CHECK_CUDA_ERROR(cudaFree(A_d.colIdx));
    CHECK_CUDA_ERROR(cudaFree(A_d.value));
    CHECK_CUDA_ERROR(cudaFree(B_d));
    CHECK_CUDA_ERROR(cudaFree(C_d));
    free(A_h.rowPtrs);
    free(A_h.colIdx);
    free(A_h.value);
    CHECK_CUDA_ERROR(cudaFreeHost(B_h));
    CHECK_CUDA_ERROR(cudaFreeHost(C_h));
}

} // namespace cuspmm

int main(int argc, char* argv[]) {
    std::string filePath = argv[1];
    SparseMatrix<float>* matrix = loadCSR<float>(filePath);

    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    runEngine<float>(matrix, abs_tol, rel_tol);

    return 0;
}