#include "formats/sparse_bsr.hpp"

namespace cuspmm {
template <typename T>
SparseMatrixBSR<T>::SparseMatrixBSR() : SparseMatrix<T>() {
    this->blockRowSize = 0;
    this->blockColSize = 0;
    this->numBlocks = 0;
    this->numBlockRows = 0;
    this->numBlockRows = 0;
    this->numElements = 0;
    this->blockRowPtrs = nullptr;
    this->blockColIdxs = nullptr;
}

template <typename T>
SparseMatrixBSR<T>::SparseMatrixBSR(std::string filePath) {
    this->onDevice = false;
    this->blockRowPtrs = nullptr;
    this->blockColIdxs = nullptr;

    std::ifstream inputFile(filePath);
    std::string line;

    if (!inputFile.is_open()) {
        std::cerr << "File " << filePath << "doesn't exist!" << std::endl;
        throw std::runtime_error(NULL);
    }

    inputFile >> this->numRows >> this->numCols >> this->numNonZero >>
        this->blockRowSize >> this->blockColSize >> this->numBlocks;
    // Calculate metrics
    this->numBlockRows = this->numRows / this->blockRowSize;
    this->numElements =
        this->numBlocks * this->blockRowSize * this->blockColSize;

    std::getline(inputFile, line); // Discard the line

    this->allocateSpace(false);

    // Read row ptrs
    std::getline(inputFile, line);
    std::istringstream iss(line);
    for (int i = 0; i <= this->numBlockRows; i++) {
        iss >> this->blockRowPtrs[i];
    }

    // Read column index
    std::getline(inputFile, line);
    iss.str(line);
    iss.clear();
    for (int i = 0; i <= this->numBlocks; i++) {
        iss >> this->blockColIdxs[i];
    }

    // Read data
    for (int i = 0; i < this->numElements; i++) {
        inputFile >> this->data[i];
    }
}

template <typename T>
SparseMatrixBSR<T>::SparseMatrixBSR(Matrix::metadataType numRows,
                                    Matrix::metadataType numCols,
                                    Matrix::metadataType numNonZero,
                                    Matrix::metadataType blockRowSize,
                                    Matrix::metadataType blockColSize,
                                    Matrix::metadataType numBlocks,
                                    bool onDevice) {
    this->blockRowSize = blockRowSize;
    this->blockColSize = blockColSize;
    this->numBlocks = numBlocks;
    this->blockRowPtrs = nullptr;
    this->blockColIdxs = nullptr;

    // Calculate depended numbers
    this->numBlockRows = numBlockRows;
    this->numElements = numBlocks * blockRowSize * blockColSize;

    this->numRows = numRows;
    this->numCols = numCols;
    this->numNonZero = numNonZero;
    this->onDevice = onDevice;

    this->allocateSpace(onDevice);
    this->assertCheck();
}

template <typename T>
SparseMatrixBSR<T>::SparseMatrixBSR(SparseMatrixBSR<T> *target, bool onDevice) {
    this->blockRowSize = target->blockRowSize;
    this->blockColSize = target->blockColSize;
    this->numBlocks = target->numBlocks;
    this->blockRowPtrs = nullptr;
    this->blockColIdxs = nullptr;

    // Calculate depended numbers
    this->numBlockRows = target->numBlockRows;
    this->numElements = target->numElements;

    this->numRows = target->numRows;
    this->numCols = target->numCols;
    this->numNonZero = target->numNonZero;
    this->onDevice = onDevice;

    this->allocateSpace(this->onDevice);
    this->copyData(target, this->onDevice);
    this->assertCheck();
}

template <typename T> SparseMatrixBSR<T>::~SparseMatrixBSR() {
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

template <typename T>
bool SparseMatrixBSR<T>::copyData(SparseMatrixBSR<T> *source, bool onDevice) {
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

    cudaCheckError(cudaMemcpy(
        this->blockRowPtrs, source->blockRowPtrs,
        (this->numBlockRows + 1) * sizeof(Matrix::metadataType), type));
    cudaCheckError(cudaMemcpy(this->blockColIdxs, source->blockColIdxs,
                              this->numBlocks * sizeof(Matrix::metadataType),
                              type));
    cudaCheckError(cudaMemcpy(this->data, source->data,
                              this->numElements * sizeof(T), type));

    return true;
}

template <typename T> SparseMatrixBSR<T> *SparseMatrixBSR<T>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixBSR<T> *newMatrix = new SparseMatrixBSR<T>(this, true);

    return newMatrix;
}

template <typename T> void SparseMatrixBSR<T>::assertCheck() {
    assert(this->numRows % this->blockRowSize == 0);
    assert(this->numCols % this->blockColSize == 0);
}

template <typename T>
void SparseMatrixBSR<T>::assertSameShape(SparseMatrixBSR<T> *target) {
    assert(this->blockRowSize == target->blockRowSize &&
           this->blockColSize == target->blockColSize &&
           this->numBlocks == target->numBlocks &&
           this->numBlockRows == target->numBlockRows &&
           this->numRows == target->numRows &&
           this->numCols == target->numCols &&
           this->numNonZero == target->numNonZero);
}

template <typename T> bool SparseMatrixBSR<T>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    if (onDevice) {
        cudaCheckError(cudaMalloc(&this->data, this->numElements * sizeof(T)));
        cudaCheckError(
            cudaMemset(this->data, 0, this->numElements * sizeof(T)));

        cudaCheckError(
            cudaMalloc(&this->blockRowPtrs, (this->numBlockRows + 1) *
                                                sizeof(Matrix::metadataType)));
        cudaCheckError(cudaMemset(this->blockRowPtrs, 0,
                                  (this->numBlockRows + 1) *
                                      sizeof(Matrix::metadataType)));

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

        cudaCheckError(
            cudaMallocHost(&this->blockColIdxs,
                           this->numBlocks * sizeof(Matrix::metadataType)));
        std::memset(this->blockColIdxs, 0,
                    this->numBlocks * sizeof(Matrix::metadataType));
    }

    return true;
}

template <typename T>
SparseMatrixBSR<T> *
SparseMatrixBSR<T>::fromDense(DenseMatrix<T> *dense,
                              Matrix::metadataType blockRowSize,
                              Matrix::metadataType blockColSize) {
    throw std::runtime_error("Not Implemented");
    const T zero = 0;
    assert(dense->numRows % blockRowSize == 0 &&
           dense->numCols % blockColSize == 0);

    using mt = Matrix::metadataType;
    mt numBlockRows = dense->numRows / blockRowSize;
    mt numBlockCols = dense->numCols / blockColSize;

    for (mt blockRowIdx = 0; blockRowIdx < numBlockRows; blockRowIdx++) {
        for (mt blockColIdx = 0; blockColIdx < numBlockCols; blockColIdx++) {
            // 1. Check if the block have non-zero elements
            mt blockRowBase = blockRowSize * blockRowIdx;
            mt blockColBase = blockColSize * blockColIdx;
            bool haveNonZero = false;
            for (int i = 0; i < blockRowSize; i++) {
                for (int j = 0; j < blockColSize; j++) {
                    if (dense->data[RowMjIdx(blockRowBase + i, blockColBase + j,
                                             dense->numCols)] != zero) {
                        haveNonZero = true;
                        break;
                    }
                }
                if (haveNonZero)
                    break;
            }

            if (!haveNonZero)
                continue;

            // TODO: Process the block
            ;
        }
    }

    return nullptr;
}

template <typename T> DenseMatrix<T> *SparseMatrixBSR<T>::toDense() {
    assert(!this->onDevice);

    using mt = Matrix::metadataType;

    DenseMatrix<T> *dm =
        new DenseMatrix<T>(this->numRows, this->numCols, false);

    for (mt blockRow = 0; blockRow < this->numBlockRows; blockRow++) {
        mt blockRowStart = this->blockRowPtrs[blockRow];
        mt blockRowEnd = this->blockRowPtrs[blockRow + 1];
        for (mt blockIdx = blockRowStart; blockIdx < blockRowEnd; blockIdx++) {
            mt blockCol = this->blockColIdxs[blockIdx];
            T *blockData = this->data +
                           (this->blockRowSize * this->blockColSize * blockIdx);

            mt denseRowStart = blockRow * this->blockRowSize;
            mt denseColStart = blockCol * this->blockColSize;
            for (mt r = 0; r < this->blockRowSize; r++) {
                for (mt c = 0; c < this->blockColSize; c++) {
                    dm->data[RowMjIdx((denseRowStart + r), (denseColStart + c),
                                      (dm->numCols))] =
                        blockData[RowMjIdx(r, c, this->blockRowSize)];
                }
            }
        }
    }

    return dm;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, SparseMatrixBSR<T> &m) {
    throw std::runtime_error("Not implemented");
    out << m.numRows << ' ' << m.numCols << ' ' << m.numNonZero << ' '
        << m.blockRowSize << ' ' << m.numBlocks << std::endl;
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

template class SparseMatrixBSR<float>;
template class SparseMatrixBSR<double>;
} // namespace cuspmm