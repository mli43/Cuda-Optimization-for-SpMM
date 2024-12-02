#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include "cuda_utils.hpp"
#include "format.hpp"
#include "cuda_runtime.h"

namespace cuspmm {

template<typename  T>
SparseMatrixCSR<T>* loadCSR(std::string& filePath) {
    std::ifstream inputFile(filePath);

    if (!inputFile.is_open()) {
        std::cerr << "File " << filePath << "doesn't exist!" << std::endl;
        return NULL;
    }

    auto matrix = new SparseMatrixCSR<T>;
    inputFile >> matrix->numRows >> matrix->numCols >> matrix->numNonZero;

    std::string line;

    // Read row ptrs
    cudaCheckError(cudaMallocHost(&matrix->rowPtrs, sizeof(SparseMatrixCSR<T>::metadataType) * (matrix->numRows + 1)))
    std::getline(inputFile, line);
    std::istringstream iss(line);
    for (int i = 0; i <= matrix->numRows; i++) {
        iss >> matrix->rowPtrs[i];
    }

    // Read column index
    cudaCheckError(cudaMallocHost(&matrix->colIdxs, sizeof(SparseMatrixCSR<T>::metadataType) * matrix->numNonZero))
    std::getline(inputFile, line);
    iss.str(line);
    iss.clear();
    for (int i = 0; i <= matrix->numRows; i++) {
        iss >> matrix->colIdxs[i];
    }

    // Read column index
    cudaCheckError(cudaMallocHost(&matrix->data, sizeof(SparseMatrixCSR<T>::metadataType) * matrix->numNonZero))
    std::getline(inputFile, line);
    iss.str(line);
    iss.clear();
    for (int i = 0; i < matrix->numNonZero; i++) {
        iss >> matrix->data[i];
    }

    return matrix;
}

}