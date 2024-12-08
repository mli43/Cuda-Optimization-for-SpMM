#pragma once

#include "cuda_runtime.h"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cuspmm {

template <typename T> class SparseMatrixCSR : public SparseMatrix<T> {
  public:
    Matrix::metadataType *rowPtrs;
    Matrix::metadataType *colIdxs;

    SparseMatrixCSR() : SparseMatrix<T>() {
        this->rowPtrs = nullptr;
        this->colIdxs = nullptr;
    }

    SparseMatrixCSR(std::string filePath) : rowPtrs(nullptr), colIdxs(nullptr) {
        this->onDevice = false;

        std::ifstream inputFile(filePath);
        std::string line;

        if (!inputFile.is_open()) {
            std::cerr << "File " << filePath << "doesn't exist!" << std::endl;
            throw std::runtime_error(NULL);
        }

        inputFile >> this->numRows >> this->numCols >> this->numNonZero;
        std::getline(inputFile, line); // Discard the line

        this->allocateSpace(false);

        // Read row ptrs
        std::getline(inputFile, line);
        std::istringstream iss(line);
        for (int i = 0; i <= this->numRows; i++) {
            iss >> this->rowPtrs[i];
        }

        // Read column index
        std::getline(inputFile, line);
        iss.str(line);
        iss.clear();
        for (int i = 0; i <= this->numNonZero; i++) {
            iss >> this->colIdxs[i];
        }

        // Read data
        std::getline(inputFile, line);
        iss.str(line);
        iss.clear();
        for (int i = 0; i < this->numNonZero; i++) {
            iss >> this->data[i];
        }
    }

    SparseMatrixCSR(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, bool onDevice)
        : rowPtrs(nullptr), colIdxs(nullptr) {
        this->numRows = numRows;
        this->numCols = numCols;
        this->numNonZero = numNonZero;
        this->onDevice = onDevice;
        this->allocateSpace(onDevice);
    }

    ~SparseMatrixCSR() {
        if (this->rowPtrs != nullptr) {
            if (this->onDevice) {
                cudaCheckError(cudaFree(this->rowPtrs));
            } else {
                cudaCheckError(cudaFreeHost(this->rowPtrs));
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

    SparseMatrixCSR<T> *copy2Device() {
        assert(this->onDevice == false);
        assert(this->data != nullptr);

        SparseMatrixCSR<T> *newMatrix = new SparseMatrixCSR<T>(
            this->numRows, this->numCols, this->numNonZero, true);

        cudaCheckError(
            cudaMemcpy(newMatrix->rowPtrs, this->rowPtrs,
                       (this->numRows + 1) * sizeof(Matrix::metadataType),
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
        if (onDevice) {
            cudaCheckError(
                cudaMalloc(&this->data, this->numNonZero * sizeof(T)));
            cudaCheckError(
                cudaMalloc(&this->rowPtrs,
                           (this->numRows + 1) * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMalloc(&this->colIdxs,
                           this->numNonZero * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMemset(this->data, 0, this->numNonZero * sizeof(T)));
            cudaCheckError(
                cudaMemset(this->rowPtrs, 0,
                           (this->numRows + 1) * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMemset(this->colIdxs, 0,
                           this->numNonZero * sizeof(Matrix::metadataType)));
        } else {
            cudaCheckError(
                cudaMallocHost(&this->data, this->numNonZero * sizeof(T)));
            cudaCheckError(cudaMallocHost(&this->rowPtrs,
                                          (this->numRows + 1) *
                                              sizeof(Matrix::metadataType)));
            cudaCheckError(cudaMallocHost(&this->colIdxs,
                                          this->numNonZero *
                                              sizeof(Matrix::metadataType)));
            std::memset(this->data, 0, this->numNonZero * sizeof(T));
            std::memset(this->rowPtrs, 0,
                        (this->numRows + 1) * sizeof(Matrix::metadataType));
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

        for (mt r = 0; r < this->numRows; r++) {
            mt row_start = this->rowPtrs[r];
            mt row_end = this->rowPtrs[r + 1];
            for (mt idx = row_start; idx < row_end; idx++) {
                mt c = this->colIdxs[idx];
                dm->data[r * dm->numCols + c] = this->data[idx];
            }
        }
        return dm;
    }

    friend std::ostream &operator<<(std::ostream &out, SparseMatrixCSR<T> &m) {
        out << m.numRows << ' ' << m.numCols << ' ' << m.numNonZero
            << std::endl;
        for (size_t i = 0; i < m.numRows; i++) {
            out << m.rowPtrs[i] << ' ';
        }
        out << std::endl;

        for (size_t i = 0; i < m.numNonZero; i++) {
            out << m.colIdxs[i] << ' ';
        }
        out << std::endl;

        for (size_t i = 0; i < m.numNonZero; i++) {
            out << m.data[i] << ' ';
        }
        out << std::endl;

        return out;
    }
};

} // namespace cuspmm
