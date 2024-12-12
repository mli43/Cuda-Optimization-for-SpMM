#include "formats/sparse_ell.hpp"

namespace cuspmm {

template <typename T>
SparseMatrixELL<T>::SparseMatrixELL() : SparseMatrix<T>() {
    this->colIdxs = nullptr;
    this->maxRowNnz = 0;
}

template <typename T>
SparseMatrixELL<T>::SparseMatrixELL(std::string colindPath,
                                    std::string valuesPath) {
    this->colIdxs = nullptr;
    this->onDevice = false;

    std::ifstream colindFile(colindPath);
    std::string line_colind;

    std::ifstream valuesFile(valuesPath);
    std::string line_values;

    if (!colindFile.is_open()) {
        std::cerr << "File " << colindPath << "doesn't exist!" << std::endl;
        throw std::runtime_error(NULL);
    }

    if (!valuesFile.is_open()) {
        std::cerr << "File " << valuesPath << "doesn't exist!" << std::endl;
        throw std::runtime_error(NULL);
    }

    colindFile >> this->numRows >> this->numCols >> this->numNonZero >>
        this->maxRowNnz;
    constexpr auto max_size = std::numeric_limits<std::streamsize>::max();
    colindFile.ignore(max_size, '\n');

    this->allocateSpace(false);

    // Read col indexes
    for (size_t row = 0; row < this->numRows; row++) {
        for (size_t i = 0; i < this->maxRowNnz; i++) {
            colindFile >> this->colIdxs[row * this->maxRowNnz + i];
        }
    }

    // Read values
    for (size_t row = 0; row < this->numRows; row++) {
        for (size_t i = 0; i < this->maxRowNnz; i++) {
            valuesFile >> this->data[row * this->maxRowNnz + i];
        }
    }

    colindFile.close();
    valuesFile.close();
}

template <typename T>
SparseMatrixELL<T>::SparseMatrixELL(Matrix::metadataType numRows,
                                    Matrix::metadataType numCols,
                                    Matrix::metadataType numNonZero,
                                    Matrix::metadataType maxRowNnz,
                                    bool onDevice) {
    this->numRows = numRows;
    this->numCols = numCols;
    this->numNonZero = numNonZero;
    this->onDevice = onDevice;
    this->maxRowNnz = maxRowNnz;
    this->colIdxs = nullptr;
    this->data = nullptr;
    this->allocateSpace(onDevice);
}

template <typename T> SparseMatrixELL<T>::~SparseMatrixELL() {
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

template <typename T> bool SparseMatrixELL<T>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    assert(this->colIdxs == nullptr);
    if (onDevice) {

        cudaCheckError(cudaMalloc(&this->data,
                                  this->numRows * this->maxRowNnz * sizeof(T)));
        cudaCheckError(
            cudaMalloc(&this->colIdxs, this->numRows * this->maxRowNnz *
                                           sizeof(Matrix::metadataType)));
        cudaCheckError(cudaMemset(this->data, 0,
                                  this->numRows * this->maxRowNnz * sizeof(T)));
        cudaCheckError(cudaMemset(this->colIdxs, 0,
                                  this->numRows * this->maxRowNnz *
                                      sizeof(Matrix::metadataType)));
    } else {
        cudaCheckError(cudaMallocHost(
            &this->data, this->numRows * this->maxRowNnz * sizeof(T)));
        cudaCheckError(
            cudaMallocHost(&this->colIdxs, this->numRows * this->maxRowNnz *
                                               sizeof(Matrix::metadataType)));
        std::memset(this->data, 0, this->numRows * this->maxRowNnz * sizeof(T));
        std::memset(this->colIdxs, 0,
                    this->numRows * this->maxRowNnz *
                        sizeof(Matrix::metadataType));
    }

    return true;
}

template <typename T> SparseMatrixELL<T> *SparseMatrixELL<T>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixELL<T> *newMatrix = new SparseMatrixELL<T>(
        this->numRows, this->numCols, this->numNonZero, this->maxRowNnz, true);

    cudaCheckError(cudaMemcpy(newMatrix->colIdxs, this->colIdxs,
                              this->numRows * this->maxRowNnz *
                                  sizeof(Matrix::metadataType),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                              this->numRows * this->maxRowNnz * sizeof(T),
                              cudaMemcpyHostToDevice));

    return newMatrix;
}

template <typename T> DenseMatrix<T> *SparseMatrixELL<T>::toDense() {
    assert(!this->onDevice);

    using mt = Matrix::metadataType;

    DenseMatrix<T> *dm =
        new DenseMatrix<T>(this->numRows, this->numCols, false);

    for (size_t row = 0; row < this->numRows; row++) {
        size_t base = row * this->maxRowNnz;
        for (size_t colind = 0; colind < this->maxRowNnz; colind++) {
            int col = this->colIdxs[base + colind];
            if (col >= 0) {
                dm->data[row * dm->numCols + col] = this->data[base + colind];
            }
        }
    }

    return dm;
}

template class SparseMatrixELL<float>;
template class SparseMatrixELL<double>;

} // namespace cuspmm