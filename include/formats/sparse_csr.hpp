#pragma once

#include "formats/matrix.hpp"
#include "cuda_runtime.h"

namespace cuspmm {

template<typename T>
class SparseMatrixCSR : public SparseMatrix<T>{
public:
    Matrix::metadataType* rowPtrs;
    Matrix::metadataType* colIdxs;

    SparseMatrixCSR() : SparseMatrix<T>(){
        this->rowPtrs = nullptr;
        this->colIdxs = nullptr;
    }

    void toDeivce(SparseMatrixCSR<T>* hm) {
        cudaCheckError(cudaMalloc(&this->rowPtrs, (hm->numRows + 1) * sizeof(SparseMatrixCSR<T>::metadataType)));
        cudaCheckError(cudaMalloc(&this->colIdxs, hm->numNonZero * sizeof(SparseMatrixCSR<T>::metadataType)));
        cudaCheckError(cudaMalloc(&this->data, hm->numNonZero * sizeof(T)));

        this->numRows = hm->numRows;
        this->numCols = hm->numCols;
        this->numNonZero = hm->numNonZero;

        cudaCheckError(cudaMemcpy(this->rowPtrs, hm->rowPtrs,
                                    (hm->numRows + 1) * sizeof(SparseMatrixCSR<T>::metadataType),
                                    cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(this->colIdxs, hm->colIdxs,
                                    hm->num_nonzero * sizeof(SparseMatrixCSR<T>::metadataType),
                                    cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(this->data, hm->data, hm->numNonZero * sizeof(T),
                                    cudaMemcpyHostToDevice));
    }
};

}
