#include "formats/sparse_ell.hpp"

namespace cuspmm {

template <typename T>
SparseMatrixELL<T>::SparseMatrixELL() : SparseMatrix<T>() {
    this->rowIdxs = nullptr;
    this->maxColNnz = 0;
}

template <typename T>
SparseMatrixELL<T>::SparseMatrixELL(std::string rowindPath,
                                    std::string valuesPath) {
    this->rowIdxs = nullptr;
    this->onDevice = false;

    std::ifstream rowindFile(rowindPath);

    std::ifstream valuesFile(valuesPath);

    if (!rowindFile.is_open()) {
        std::cerr << "File " << rowindPath << "doesn't exist!" << std::endl;
        throw std::runtime_error(NULL);
    }

    if (!valuesFile.is_open()) {
        std::cerr << "File " << valuesPath << "doesn't exist!" << std::endl;
        throw std::runtime_error(NULL);
    }

    rowindFile >> this->numRows >> this->numCols >> this->numNonZero >>
        this->maxColNnz;
    constexpr auto max_size = std::numeric_limits<std::streamsize>::max();
    rowindFile.ignore(max_size, '\n');

    this->allocateSpace(false);

    // Read col indexes
    for (size_t col = 0; col < this->numCols; col++) {
        for (size_t i = 0; i < this->maxColNnz; i++) {
            rowindFile >> this->rowIdxs[col * this->maxColNnz + i];
        }
    }

    // Read values
    for (size_t col = 0; col < this->numCols; col++) {
        for (size_t i = 0; i < this->maxColNnz; i++) {
            valuesFile >> this->data[col * this->maxColNnz + i];
        }
    }

    rowindFile.close();
    valuesFile.close();
}

template <typename T>
SparseMatrixELL<T>::SparseMatrixELL(Matrix::metadataType numRows,
                                    Matrix::metadataType numCols,
                                    Matrix::metadataType numNonZero,
                                    Matrix::metadataType maxColNnz,
                                    bool onDevice) {
    this->numRows = numRows;
    this->numCols = numCols;
    this->numNonZero = numNonZero;
    this->onDevice = onDevice;
    this->maxColNnz = maxColNnz;
    this->rowIdxs = nullptr;
    this->data = nullptr;
    this->allocateSpace(onDevice);
}

template <typename T> SparseMatrixELL<T>::~SparseMatrixELL() {
    if (this->rowIdxs != nullptr) {
        if (this->onDevice) {
            cudaCheckError(cudaFree(this->rowIdxs));
        } else {
            cudaCheckError(cudaFreeHost(this->rowIdxs));
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
void SparseMatrixELL<T>::setCusparseSpMatDesc(cusparseSpMatDescr_t *matDescP) {
    cudaDataType dt;
    if constexpr (std::is_same<T, half>::value) {
        dt = CUDA_R_16F;
    } else if constexpr (std::is_same<T, float>::value) {
        dt = CUDA_R_32F;
    } else if constexpr (std::is_same<T, double>::value) {
        dt = CUDA_R_64F;
    }
    assertTypes3(T, half, float, double);

    // FIXME:
    throw std::runtime_error("not implemented");
}

template <typename T>
cusparseSpMMAlg_t SparseMatrixELL<T>::getCusparseAlg() {
    return CUSPARSE_SPMM_ALG_DEFAULT;
}

template <typename T> bool SparseMatrixELL<T>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    assert(this->rowIdxs == nullptr);
    if (onDevice) {

        cudaCheckError(cudaMalloc(&this->data,
                                  this->numCols * this->maxColNnz * sizeof(T)));
        cudaCheckError(
            cudaMalloc(&this->rowIdxs, this->numCols * this->maxColNnz *
                                           sizeof(Matrix::metadataType)));
        cudaCheckError(cudaMemset(this->data, 0,
                                  this->numCols * this->maxColNnz * sizeof(T)));
        cudaCheckError(cudaMemset(this->rowIdxs, 0,
                                  this->numCols * this->maxColNnz *
                                      sizeof(Matrix::metadataType)));
    } else {
        cudaCheckError(cudaMallocHost(
            &this->data, this->numCols * this->maxColNnz * sizeof(T)));
        cudaCheckError(
            cudaMallocHost(&this->rowIdxs, this->numCols * this->maxColNnz *
                                               sizeof(Matrix::metadataType)));
        std::memset(this->data, 0, this->numCols * this->maxColNnz * sizeof(T));
        std::memset(this->rowIdxs, 0,
                    this->numCols * this->maxColNnz *
                        sizeof(Matrix::metadataType));
    }

    return true;
}

template <typename T> SparseMatrixELL<T> *SparseMatrixELL<T>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixELL<T> *newMatrix = new SparseMatrixELL<T>(
        this->numRows, this->numCols, this->numNonZero, this->maxColNnz, true);

    cudaCheckError(cudaMemcpy(newMatrix->rowIdxs, this->rowIdxs,
                              this->numCols* this->maxColNnz *
                                  sizeof(Matrix::metadataType),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                              this->numCols * this->maxColNnz * sizeof(T),
                              cudaMemcpyHostToDevice));

    return newMatrix;
}

template <typename T> DenseMatrix<T> *SparseMatrixELL<T>::toDense() {
    assert(!this->onDevice);

    using mt = Matrix::metadataType;

    DenseMatrix<T> *dm =
        new DenseMatrix<T>(this->numRows, this->numCols, false);

    for (size_t col = 0; col < this->numCols; col++) {
        size_t base = col * this->maxColNnz;
        for (size_t rowind = 0; rowind < this->maxColNnz; rowind++) {
            int row = this->rowIdxs[base + rowind];
            if (row >= 0) {
                dm->data[row * dm->numCols + col] = this->data[base + rowind];
            }
        }
    }

    return dm;
}

template class SparseMatrixELL<float>;
template class SparseMatrixELL<double>;

} // namespace cuspmm
