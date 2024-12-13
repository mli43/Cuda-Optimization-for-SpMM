#include "formats/sparse_coo.hpp"

namespace cuspmm {

template <typename T>
SparseMatrixCOO<T>::SparseMatrixCOO() : SparseMatrix<T>() {
    this->rowIdxs = nullptr;
    this->colIdxs = nullptr;
}

template <typename T>
SparseMatrixCOO<T>::SparseMatrixCOO(std::string filePath) {
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

template <typename T>
SparseMatrixCOO<T>::SparseMatrixCOO(Matrix::metadataType numRows,
                                    Matrix::metadataType numCols,
                                    Matrix::metadataType numNonZero,
                                    bool onDevice)
    : rowIdxs(nullptr), colIdxs(nullptr) {
    this->numRows = numRows;
    this->numCols = numCols;
    this->numNonZero = numNonZero;
    this->onDevice = onDevice;
    this->allocateSpace(onDevice);
}

template <typename T> SparseMatrixCOO<T>::~SparseMatrixCOO() {
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

template <typename T>
void SparseMatrixCOO<T>::setCusparseSpMatDesc(cusparseSpMatDescr_t *matDescP) {
    cudaDataType dt;
    if constexpr (std::is_same<T, half>::value) {
        dt = CUDA_R_16F;
    } else if constexpr (std::is_same<T, float>::value) {
        dt = CUDA_R_32F;
    } else if constexpr (std::is_same<T, double>::value) {
        dt = CUDA_R_64F;
    }
    assertTypes3(T, half, float, double);

    CHECK_CUSPARSE(cusparseCreateCoo(
        matDescP, this->numRows, this->numCols, this->numNonZero, this->rowIdxs,
        this->colIdxs, this->data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        dt));
}

template <typename T>
cusparseSpMMAlg_t SparseMatrixCOO<T>::getCusparseAlg() {
    return CUSPARSE_SPMM_COO_ALG4;
}

template <typename T> SparseMatrixCOO<T> *SparseMatrixCOO<T>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixCOO<T> *newMatrix = new SparseMatrixCOO<T>(
        this->numRows, this->numCols, this->numNonZero, true);

    cudaCheckError(cudaMemcpy(newMatrix->rowIdxs, this->rowIdxs,
                              this->numNonZero * sizeof(Matrix::metadataType),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->colIdxs, this->colIdxs,
                              this->numNonZero * sizeof(Matrix::metadataType),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                              this->numNonZero * sizeof(T),
                              cudaMemcpyHostToDevice));
    return newMatrix;
}

template <typename T> bool SparseMatrixCOO<T>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    assert(this->rowIdxs == nullptr);
    assert(this->colIdxs == nullptr);
    if (onDevice) {
        cudaCheckError(cudaMalloc(&this->data, this->numNonZero * sizeof(T)));
        cudaCheckError(cudaMalloc(
            &this->rowIdxs, this->numNonZero * sizeof(Matrix::metadataType)));
        cudaCheckError(cudaMalloc(
            &this->colIdxs, this->numNonZero * sizeof(Matrix::metadataType)));
        cudaCheckError(cudaMemset(this->data, 0, this->numNonZero * sizeof(T)));
        cudaCheckError(cudaMemset(
            this->rowIdxs, 0, this->numNonZero * sizeof(Matrix::metadataType)));
        cudaCheckError(cudaMemset(
            this->colIdxs, 0, this->numNonZero * sizeof(Matrix::metadataType)));
    } else {
        cudaCheckError(
            cudaMallocHost(&this->data, this->numNonZero * sizeof(T)));
        cudaCheckError(cudaMallocHost(
            &this->rowIdxs, this->numNonZero * sizeof(Matrix::metadataType)));
        cudaCheckError(cudaMallocHost(
            &this->colIdxs, this->numNonZero * sizeof(Matrix::metadataType)));
        std::memset(this->data, 0, this->numNonZero * sizeof(T));
        std::memset(this->rowIdxs, 0,
                    this->numNonZero * sizeof(Matrix::metadataType));
        std::memset(this->colIdxs, 0,
                    this->numNonZero * sizeof(Matrix::metadataType));
    }

    return true;
}

template <typename T> DenseMatrix<T> *SparseMatrixCOO<T>::toDense() {
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

template <typename T>
std::ostream &operator<<(std::ostream &out, SparseMatrixCOO<T> &m) {
    out << m.numRows << ' ' << m.numCols << ' ' << m.numNonZero << std::endl;
    for (size_t i = 0; i < m->numNonZero; i++) {
        std::cout << m.rowIdxs[i] << ' ' << m.colIdxs[i] << ' ' << m.data[i]
                  << std::endl;
    }

    return out;
}

template class SparseMatrixCOO<float>;
template class SparseMatrixCOO<double>;
} // namespace cuspmm