#pragma once

#include "cuda_utils.hpp"
#include "formats/matrix.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstring>

namespace cuspmm {

template <typename T> class DenseMatrix : public Matrix {
  public:
    T *data;
    DenseMatrix() : Matrix() { this->data = nullptr; }

    DenseMatrix(std::string filePath) {
        std::ifstream inputFile(filePath);
        std::string line;

        if (!inputFile.is_open()) {
            std::cerr << "File " << filePath << "doesn't exist!" << std::endl;
            throw std::runtime_error(NULL);
        }

        inputFile >> this->numRows >> this->numCols;
        std::getline(inputFile, line); // Discard the header line


        // Read column index
        uint32_t total = this->numRows * this->numCols;
        cudaCheckError(
            cudaMallocHost(&this->data, sizeof(Matrix::metadataType) * total));

        for (int i = 0; i < this->numRows; i++) {
            std::getline(inputFile, line);
            std::istringstream iss(line);
            for (int j = 0; j < this->numCols; j++) {
                iss>> this->data[i * this->numCols + j];
            }
        }

        this->onDevice = false;
    }

    DenseMatrix(Matrix::metadataType numRows,
                       Matrix::metadataType numCols, bool onDevice) {
        this->numRows = numRows;
        this->numCols = numCols;
        this->onDevice = onDevice;
        this->data = nullptr;
        this->allocateSpace(onDevice);
    }

    DenseMatrix<T>* copy2Device() {
        assert(this->onDevice == false);
        assert(this->data != nullptr);

        DenseMatrix<T>* newMatrix = new DenseMatrix<T>(this->numRows, this->numCols, true);

        uint64_t totalSize = sizeof(T) * (this->numRows * this->numCols);;
        cudaCheckError(cudaMemcpy(newMatrix->data, this->data, totalSize, cudaMemcpyHostToDevice));

        return newMatrix;
    }

    DenseMatrix<T>* copy2Host() {
        assert(this->onDevice == true);
        assert(this->data != nullptr);

        DenseMatrix<T>* newMatrix = new DenseMatrix<T>(this->numRows, this->numCols, false);

        uint64_t totalSize = sizeof(T) * (this->numRows * this->numCols);;
        cudaCheckError(cudaMemcpy(newMatrix->data, this->data, totalSize, cudaMemcpyDeviceToHost));

        return newMatrix;
    }

    bool save2File(std::string filePath) {
        using mt = Matrix::metadataType;
        assert(!this->onDevice);

        std::ofstream outputFile(filePath);
        if (!outputFile.is_open()) {
            std::cerr << "Cannot open output file " << filePath << std::endl;
            return false;
        }

        outputFile << this->numRows << ' ' << this->numCols << std::endl;
        for (mt r = 0; r < this->numRows; r++) {
            for (mt c = 0; c < this->numCols; c++) {
                outputFile << this->data[r * this->numCols + c] << ' ';
            }
            outputFile << std::endl;
        }

        return true;
    }

    bool allocateSpace(bool onDevice) {
        assert(this->data == nullptr);

        size_t totalSize = this->numRows * this->numCols * sizeof(T);
        if (onDevice) {
            cudaCheckError(cudaMalloc(
                &this->data, totalSize));
            cudaCheckError(cudaMemset(this->data, 0, totalSize));
        } else {
            cudaCheckError(cudaMallocHost(
                &this->data, totalSize));
            std::memset(this->data, 0, totalSize);
        }

        return true;
    }
};

} // namespace cuspmm
