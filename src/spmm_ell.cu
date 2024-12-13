#include "cuda_utils.hpp"
#include "spmm_ell.hpp"
#include "torch/torch.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <chrono>

#define BLOCKSIZE 1024

namespace cuspmm {

template <typename T, typename MT, typename AccT>
__global__ void spmmELLK1(MT aNumRows, MT aNumCols, MT aNumNonZero, MT aMaxColNnz,
                                    MT *rowIdxs, T* aData, 
                                    MT bNumRows, MT bNumCols, T* bData,
                                    T* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t numValues = aNumCols * aMaxColNnz;

    if (idx < numValues) {
        int col = idx / aMaxColNnz;
        int row = rowIdxs[idx];
        float value = aData[idx];

        

        if (row>= 0) {
            for (int j = 0; j < bNumCols; j++) {
                atomicAdd(&cData[row * bNumCols + j], value * bData[col * bNumCols + j]); 
            }
        }
    }
}

template <typename T, typename MT, typename AccT>
__global__ void spmmELLK2(MT aNumRows, MT aNumCols, MT aNumNonZero, MT aMaxColNnz,
                                    MT *rowIdxs, T* aData, 
                                    MT bNumRows, MT bNumCols, T* bData,
                                    T* cData) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T bCol[BLOCKSIZE];

    size_t numValues = aNumCols * aMaxColNnz;

    int row, col;
    float value;
    row = -1;
    col = 0;
    value = 0;

    if (idx < numValues) {
        col = idx / aMaxColNnz;
        row = rowIdxs[idx];
        value = aData[idx];
    }

    for (int j = 0; j < bNumCols; j++) {
        for (int startRow = 0; startRow < bNumRows; startRow += blockDim.x) {
            if (startRow + threadIdx.x < bNumRows) {
                bCol[threadIdx.x] = bData[(startRow + threadIdx.x)* bNumCols + j];
            }
            else {
                bCol[threadIdx.x] = 0;
            }
            __syncthreads();

            if (idx < numValues) {
                if (row>= 0 && col - startRow >= 0 && col - startRow < blockDim.x) {
                    atomicAdd(&cData[row * bNumCols + j], value * bCol[col - startRow]);
                }
            }
            __syncthreads();
        }
    }
}

template <typename T, typename AccT>
DenseMatrix<T>* spmmEllDevice(SparseMatrixELL<T>* a, DenseMatrix<T>* b) {
    const size_t numValues = a->numCols * a->maxColNnz;

    dim3 block(BLOCKSIZE);
    dim3 grid((numValues + BLOCKSIZE - 1) / BLOCKSIZE);

    if (!a->onDevice || !b->onDevice) {
        std::cerr << "Device incorrect!" << std::endl; 
        return nullptr;
    }

    DenseMatrix<T>* c = new DenseMatrix<T>(a->numRows, b->numCols, true);

    spmmELLK2<T, typename SparseMatrixELL<T>::metadataType, AccT><<<grid, block>>>(
        a->numRows, a->numCols, a->numNonZero, a->maxColNnz, a->rowIdxs, a->data,
        b->numRows, b->numCols, b->data, 
        c->data
    );

    return c;
}

template <typename T>
void runEngineELL(SparseMatrixELL<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol) {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Move to device
    SparseMatrixELL<T>* da = a->copy2Device();
    DenseMatrix<T>* db = b->copy2Device();
    cudaDeviceSynchronize();
    auto copy_to_device_end = std::chrono::high_resolution_clock::now();

    // 2. Launch kernel
    auto cRes = spmmEllDevice<T, double>(da, db);
    cudaDeviceSynchronize();
    auto kernel_end = std::chrono::high_resolution_clock::now();

    auto cResCpu = cRes->copy2Host();
    cudaDeviceSynchronize();
    auto copy_to_host_end = std::chrono::high_resolution_clock::now();

    // 3. Check result
    auto cResSeq = spmmEllCpu<T, double>(a, b);
    auto seq_end = std::chrono::high_resolution_clock::now();

    // 4. Report time 
    auto copy2DeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_device_end - start);
    auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - copy_to_device_end);
    auto copy2HostTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_host_end - kernel_end);
    auto parallelTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_host_end - start);
    auto seqTime = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - copy_to_host_end);

    std::cout << "copy2DeviceTime (us):" << copy2DeviceTime.count() << ','
              << "kernelTime (us):" << kernelTime.count() << ','
              << "copy2HostTime (us):" << copy2HostTime.count() << ','
              << "parallelTime (us):" << parallelTime.count() << ','
              << "seqTime (us):" << seqTime.count() << '\n';

    cResCpu->save2File("ell_cuda.res");
    cResSeq->save2File("ell_cpu.res");

    auto denseA = a->toDense();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor taDevice = torch::from_blob(denseA->data, {denseA->numRows, denseA->numCols}, options).clone().cuda();
    torch::Tensor tbDevice = torch::from_blob(b->data, {b->numRows, b->numCols}, options).clone().cuda();
    torch::Tensor tcCpu = torch::from_blob(cResCpu->data, {cResCpu->numRows, cResCpu->numCols}, options).clone();
    torch::Tensor cResTorch = torch::matmul(taDevice, tbDevice).cpu();
    std::cout << "ell allclose: " << torch::allclose(tcCpu, cResTorch, rel_tol, abs_tol) << std::endl;

    auto denseTorch = new DenseMatrix<T>(cResCpu->numRows, cResCpu->numCols, false);
    std::memcpy(denseTorch->data, cResTorch.data_ptr<float>(), denseTorch->numRows * denseTorch->numCols * sizeof(float));
    denseTorch->save2File("ell_torch.res");
}

template void runEngineELL<float>(SparseMatrixELL<float> *a, DenseMatrix<float>* b, float abs_tol, double rel_tol);

} // namespace cuspmm
