#pragma once

#include "cuda_runtime.h"
#include "cuda_utils.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"
#include "commons.hpp"
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

template <typename T> class SparseMatrixBSR : public SparseMatrix<T> {
  public:
    Matrix::metadataType blockSize;
    Matrix::metadataType numBlocks;
    Matrix::metadataType *blockRowPtrs;
    Matrix::metadataType *blockColIdxs;
    // Can be calculated based on numbers above
    Matrix::metadataType numBlockRows;
    Matrix::metadataType numElements;

    SparseMatrixBSR() : SparseMatrix<T>() {
        this->blockSize = 0;
        this->numBlocks = 0;
        this->numBlockRows = 0;
        this->numBlockRows = 0;
        this->numElements = 0;
        this->blockRowPtrs = nullptr;
        this->blockColIdxs = nullptr;
    }

    SparseMatrixBSR(std::string filePath) {
        this->onDevice = false;
        this->blockRowPtrs = nullptr;
        this->blockColIdxs = nullptr;

        std::ifstream inputFile(filePath);
        std::string line;

        if (!inputFile.is_open()) {
            std::cerr << "File " << filePath << "doesn't exist!" << std::endl;
            throw std::runtime_error(NULL);
        }

        inputFile >> this->numRows >> this->numCols >> this->numNonZero >> this->blockSize >> this->numBlocks;
        // Calculate metrics
        this->numBlockRows = this->numRows / this->blockSize;
        this->numElements = this->numBlocks * this->blockSize * this->blockSize;

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

    SparseMatrixBSR(Matrix::metadataType numRows, Matrix::metadataType numCols,
                    Matrix::metadataType numNonZero, Matrix::metadataType blockSize, 
                    Matrix::metadataType numBlocks, bool onDevice) {
        this->blockSize = blockSize;
        this->numBlocks = numBlocks;
        this->blockRowPtrs = nullptr;
        this->blockColIdxs = nullptr;

        // Calculate depended numbers
        this->numBlockRows = numBlockRows;
        this->numElements = numBlocks * blockSize * blockSize;
        
        this->numRows = numRows;
        this->numCols = numCols;
        this->numNonZero = numNonZero;
        this->onDevice = onDevice;

        this->allocateSpace(onDevice);
        this->check();
    }

    SparseMatrixBSR(SparseMatrixBSR<T>* target, bool onDevice) {
        this->blockSize = target->blockSize;
        this->numBlocks = target->numBlocks;
        this->blockRowPtrs = nullptr;
        this->blockColIdxs = nullptr;

        // Calculate depended numbers
        this->numBlockRows = this->numBlockRows;
        this->numElements = this->numBlocks * this->blockSize * this->blockSize;

        this->numRows = target->numRows;
        this->numCols = target->numCols;
        this->numNonZero = target->numNonZero;
        this->onDevice = onDevice;

        this->allocateSpace(this->onDevice);
        this->copyData(target, this->onDevice);
        this->check();
    }

    ~SparseMatrixBSR() {
        if (this->blockRowPtrs != nullptr) {
            if (this->onDevice) {
                cudaCheckError(cudaFree(this->blockRowPtrs));
            } else {
                cudaCheckError(cudaFreeHost(this->blockRowPtrs));
            }
        }

        if (this->blockColIdxs != nullptr) {
            if (this->onDevice) {
                cudaCheckError(cudaFree(this->blockColIdxs));
            } else {
                cudaCheckError(cudaFreeHost(this->blockColIdxs));
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

    bool copyData(SparseMatrixBSR<T>* source, bool onDevice) {
        this->assertSameShape(source);
        cudaMemcpyKind type;
        if (source->onDevice && onDevice) {
            type = cudaMemcpyDeviceToDevice;
        } else if (source->onDevice && !onDevice) {
            type = cudaMemcpyDeviceToHost;
        } else if (!source->onDevice && onDevice) {
            type = cudaMemcpyHostToDevice;
        } else {
            type = cudaMemcpyHostToHost;
        }

        cudaCheckError(
            cudaMemcpy(this->blockRowPtrs, source->blockRowPtrs,
                       (this->numBlockRows + 1) * sizeof(Matrix::metadataType),
                       type));
        cudaCheckError(
            cudaMemcpy(this->blockColIdxs, source->blockColIdxs,
                       this->numBlocks * sizeof(Matrix::metadataType),
                       type));
        cudaCheckError(cudaMemcpy(this->data, source->data,
                                  this->numElements * sizeof(T),
                                  type));

        return true;
    }

    SparseMatrixBSR<T> *copy2Device() {
        assert(this->onDevice == false);
        assert(this->data != nullptr);

        SparseMatrixBSR<T> *newMatrix = new SparseMatrixBSR<T>(this, true);

        return newMatrix;
    }

    bool check() {
        assert(this->numRows % this->blockSize == 0);
        assert(this->numCols % this->blockSize == 0);
    }

    bool assertSameShape(SparseMatrixBSR<T> target) {
        assert(
        this->blockSize == target->blockSize &&
        this->numBlocks == target->numBlocks &&
        this->numBlockRows == target->numBlockRows &&
        this->numRows == target->numRows &&
        this->numCols == target->numCols &&
        this->numNonZero == target->numNonZero
        );
    }

    bool allocateSpace(bool onDevice) {
        assert(this->data == nullptr);
        if (onDevice) {
            cudaCheckError(
                cudaMalloc(&this->data, this->numElements * sizeof(T)));
            cudaCheckError(
                cudaMemset(this->data, 0, this->numElements* sizeof(T)));

            cudaCheckError(
                cudaMalloc(&this->blockRowPtrs,
                           (this->numBlockRows + 1) * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMemset(this->blockRowPtrs, 0,
                           (this->numBlockRows + 1) * sizeof(Matrix::metadataType)));

            cudaCheckError(
                cudaMalloc(&this->blockColIdxs,
                           this->numBlocks * sizeof(Matrix::metadataType)));
            cudaCheckError(
                cudaMemset(this->blockColIdxs, 0,
                           this->numBlocks * sizeof(Matrix::metadataType)));
        } else {
            cudaCheckError(
                cudaMallocHost(&this->data, this->numElements * sizeof(T)));
            std::memset(this->data, 0, this->numElements * sizeof(T));

            cudaCheckError(cudaMallocHost(&this->blockRowPtrs,
                                          (this->numBlockRows + 1) *
                                              sizeof(Matrix::metadataType)));
            std::memset(this->blockRowPtrs, 0,
                        (this->numBlockRows + 1) * sizeof(Matrix::metadataType));

            cudaCheckError(cudaMallocHost(&this->blockColIdxs,
                                          this->numBlocks *
                                              sizeof(Matrix::metadataType)));
            std::memset(this->blockColIdxs, 0,
                        this->numBlocks * sizeof(Matrix::metadataType));
        }

        return true;
    }

    SparseMatrixBSR<T>* fromDense(DenseMatrix<T>* dense) {
        throw std::runtime_error("Not implemented");


    }

    DenseMatrix<T> *toDense() {
        assert(!this->onDevice);

        using mt = Matrix::metadataType;

        DenseMatrix<T> *dm =
            new DenseMatrix<T>(this->numRows, this->numCols, false);
        
        for (mt blockRow = 0; blockRow < this->numBlockRows; blockRow++) {
            mt blockRowStart = this->blockRowPtrs[blockRow];
            mt blockRowEnd = this->blockRowPtrs[blockRow + 1];
            for (mt blockIdx = blockRowStart; blockIdx < blockRowEnd; blockIdx++) {
                mt blockCol = this->blockColIdxs[blockIdx];
                T* blockData = this->data + (this->blockSize * this->blockSize * blockIdx);

                mt denseRowStart = blockRow * this->blockSize;
                mt denseColStart = blockCol * this->blockSize;
                for (mt r = 0; r < this->blockSize; r++) {
                    for (mt c = 0; c < this->blockSize; c++) {
                        dm->data[INDEX((denseRowStart + r), (denseColStart + c), (dm->numCols))] = blockData[INDEX(r, c, this->blockSize)];
                    }
                }
            }
        }

        return dm;
    }

    friend std::ostream &operator<<(std::ostream &out, SparseMatrixBSR<T> &m) {
        throw std::runtime_error("Not implemented");
        out << m.numRows << ' ' << m.numCols << ' ' << m.numNonZero << ' ' << m.blockSize << ' ' << m.numBlocks
            << std::endl;
        for (size_t i = 0; i < m.numRows; i++) {
            out << m.blockRowPtrs[i] << ' ';
        }
        out << std::endl;

        for (size_t i = 0; i < m.numNonZero; i++) {
            out << m.blockColIdxs[i] << ' ';
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
