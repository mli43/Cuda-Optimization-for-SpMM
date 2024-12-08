#pragma once

#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace cuspmm {

template <typename T> class SparseMatrixCOO : public SparseMatrix<T> {
  public:
    Matrix::metadataType *rowIdxs;
    Matrix::metadataType *colIdxs;

    SparseMatrixCOO() : SparseMatrix<T>() {
        this->rowIdxs = nullptr;
        this->colIdxs = nullptr;
    }

    SparseMatrixCOO(std::string filePath) {
        this->rowIdxs = nullptr;
        this->colIdxs = nullptr;
        this->onDevice = false;

        std::ifstream inputFile(filePath);
        std::string line;

        if (!inputFile.is_open()) {
            std::cerr << "File " << filePath << "doesn't exist!" << std::endl;
            throw std::runtime_error(NULL);
        }

        inputFile >> this->numRows >> this->numCols >> this->numNonZero;
        constexpr auto max_size = std::numeric_limits<std::streamsize>::max();
        inputFile.ignore(max_size, '\n');

        this->allocateSpace(false);

        // Read row ptrs
        for (size_t i = 0; i < this->numNonZero; i++) {
            inputFile >> this->rowIdxs[i] >> this->colIdxs[i] >> this->data[i];
        }

        inputFile.close();
    }

    SparseMatrixCOO(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, bool onDevice)
        : rowIdxs(nullptr), colIdxs(nullptr) {
        this->numRows = numRows;
        this->numCols = numCols;
        this->numNonZero = numNonZero;
        this->onDevice = onDevice;
        this->allocateSpace(onDevice);
    }

    ~SparseMatrixCOO() {
        if (this->rowIdxs != nullptr) {
            if (this->onDevice) {
                cudaCheckError(cudaFree(this->rowIdxs));
            } else {
                cudaCheckError(cudaFreeHost(this->rowIdxs));
            }
        }

        if (this->colIdxs != nullptr) {
            if (this->onDevice) {
                cudaCheckError(cudaFree(this->colIdxs));
            } else {
                cudaCheckError(cudaFreeHost(this->colIdxs));
            }
        }

        if (this->data != nullptr) {
            if (this->onDevice) {
                cudaCheckError(cudaFree(this->data));
            } else {
                cudaCheckError(cudaFreeHost(this->data));
            }
        }
    }

    SparseMatrixCOO<T> *copy2Device() {
        assert(this->onDevice == false);
        assert(this->data != nullptr);

        SparseMatrixCOO<T> *newMatrix = new SparseMatrixCOO<T>(
            this->numRows, this->numCols, this->numNonZero, true);

        cudaCheckError(
            cudaMemcpy(newMatrix->rowIdxs, this->rowIdxs,
                       this->numNonZero * sizeof(Matrix::metadataType),
                       cudaMemcpyHostToDevice));
        cudaCheckError(
            cudaMemcpy(newMatrix->colIdxs, this->colIdxs,
                       this->numNonZero * sizeof(Matrix::metadataType),
                       cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                                  this->numNonZero * sizeof(T),
                                  cudaMemcpyHostToDevice));
        return newMatrix;
    }

    bool allocateSpace(bool onDevice) {
        assert(this->data == nullptr);
        assert(this->rowIdxs == nullptr);
        assert(this->colIdxs == nullptr);
        if (onDevice) {
            cudaCheckError(
                cudaMalloc(&this->data, this->numNonZero * sizeof(T)));
            cudaCheckError(
                cudaMalloc(&this->rowIdxs,
                           this->numNonZero * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMalloc(&this->colIdxs,
                           this->numNonZero * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMemset(this->data, 0, this->numNonZero * sizeof(T)));
            cudaCheckError(
                cudaMemset(this->rowIdxs, 0,
                           this->numNonZero * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMemset(this->colIdxs, 0,
                           this->numNonZero * sizeof(Matrix::metadataType)));
        } else {
            cudaCheckError(
                cudaMallocHost(&this->data, this->numNonZero * sizeof(T)));
            cudaCheckError(cudaMallocHost(&this->rowIdxs,
                                          this->numNonZero *
                                              sizeof(Matrix::metadataType)));
            cudaCheckError(cudaMallocHost(&this->colIdxs,
                                          this->numNonZero *
                                              sizeof(Matrix::metadataType)));
            std::memset(this->data, 0, this->numNonZero * sizeof(T));
            std::memset(this->rowIdxs, 0,
                        this->numNonZero * sizeof(Matrix::metadataType));
            std::memset(this->colIdxs, 0,
                        this->numNonZero * sizeof(Matrix::metadataType));
        }

        return true;
    }

    DenseMatrix<T> *toDense() {
        assert(!this->onDevice);

        using mt = Matrix::metadataType;

        DenseMatrix<T> *dm =
            new DenseMatrix<T>(this->numRows, this->numCols, false);

        for (size_t i = 0; i < this->numNonZero; i++) {
            mt r = this->rowIdxs[i];
            mt c = this->colIdxs[i];
            dm->data[r * dm->numCols + c] = this->data[i];
        }

        return dm;
    }

    friend std::ostream &operator<<(std::ostream &out, SparseMatrixCOO<T> &m) {
        out << m.numRows << ' ' << m.numCols << ' ' << m.numNonZero
            << std::endl;
        for (size_t i = 0; i < m->numNonZero; i++) {
            std::cout << m.rowIdxs[i] << ' ' << m.colIdxs[i] << ' ' << m.data[i]
                      << std::endl;
        }

        return out;
    }
};

} // namespace cuspmm