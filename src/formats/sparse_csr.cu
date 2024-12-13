#include "commons.hpp"
#include "formats/sparse_csr.hpp"

namespace cuspmm {
template <typename DT, typename MT>
SparseMatrixCSR<DT, MT>::SparseMatrixCSR() : SparseMatrix<DT, MT>() {
    this->rowPtrs = nullptr;
    this->colIdxs = nullptr;
}

template <typename DT, typename MT>
SparseMatrixCSR<DT, MT>::SparseMatrixCSR(std::string filePath)
    : rowPtrs(nullptr), colIdxs(nullptr) {
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

template <typename DT, typename MT>
SparseMatrixCSR<DT, MT>::SparseMatrixCSR(MT numRows,
                                    MT numCols,
                                    MT numNonZero,
                                    bool onDevice)
    : rowPtrs(nullptr), colIdxs(nullptr) {
    this->numRows = numRows;
    this->numCols = numCols;
    this->numNonZero = numNonZero;
    this->onDevice = onDevice;
    this->allocateSpace(onDevice);
}

template <typename DT, typename MT> SparseMatrixCSR<DT, MT>::~SparseMatrixCSR() {
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

template <typename DT, typename MT>
void SparseMatrixCSR<DT, MT>::setCusparseSpMatDesc(cusparseSpMatDescr_t *matDescP) {
    cudaDataType dt;
    if constexpr (std::is_same<DT, half>::value) {
        dt = CUDA_R_16F;
    } else if constexpr (std::is_same<DT, float>::value) {
        dt = CUDA_R_32F;
    } else if constexpr (std::is_same<DT, double>::value) {
        dt = CUDA_R_64F;
    }
    assertTypes3(DT, half, float, double);

    CHECK_CUSPARSE(cusparseCreateCsr(matDescP, this->numRows, this->numCols, this->numNonZero, this->rowPtrs,
                      this->colIdxs, this->data, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dt));
}


template <typename DT, typename MT> SparseMatrixCSR<DT, MT> *SparseMatrixCSR<DT, MT>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixCSR<DT, MT> *newMatrix = new SparseMatrixCSR<DT, MT>(
        this->numRows, this->numCols, this->numNonZero, true);

    cudaCheckError(
        cudaMemcpy(newMatrix->rowPtrs, this->rowPtrs,
                   (this->numRows + 1) * sizeof(DT),
                   cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->colIdxs, this->colIdxs,
                              this->numNonZero * sizeof(DT),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                              this->numNonZero * sizeof(DT),
                              cudaMemcpyHostToDevice));
    return newMatrix;
}

template <typename DT, typename MT> bool SparseMatrixCSR<DT, MT>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    if (onDevice) {
        cudaCheckError(cudaMalloc(&this->data, this->numNonZero * sizeof(DT)));
        cudaCheckError(
            cudaMalloc(&this->rowPtrs,
                       (this->numRows + 1) * sizeof(DT)));
        cudaCheckError(cudaMalloc(
            &this->colIdxs, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMemset(this->data, 0, this->numNonZero * sizeof(DT)));
        cudaCheckError(
            cudaMemset(this->rowPtrs, 0,
                       (this->numRows + 1) * sizeof(DT)));
        cudaCheckError(cudaMemset(
            this->colIdxs, 0, this->numNonZero * sizeof(DT)));
    } else {
        cudaCheckError(
            cudaMallocHost(&this->data, this->numNonZero * sizeof(DT)));
        cudaCheckError(
            cudaMallocHost(&this->rowPtrs,
                           (this->numRows + 1) * sizeof(DT)));
        cudaCheckError(cudaMallocHost(
            &this->colIdxs, this->numNonZero * sizeof(DT)));
        std::memset(this->data, 0, this->numNonZero * sizeof(DT));
        std::memset(this->rowPtrs, 0,
                    (this->numRows + 1) * sizeof(DT));
        std::memset(this->colIdxs, 0,
                    this->numNonZero * sizeof(DT));
    }

    return true;
}

template <typename DT, typename MT> DenseMatrix<DT, MT> *SparseMatrixCSR<DT, MT>::toDense() {
    assert(!this->onDevice);

    using mt = MT;

    DenseMatrix<DT, MT> *dm =
        new DenseMatrix<DT, MT>(this->numRows, this->numCols, false);

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

template <typename DT, typename MT>
cusparseSpMMAlg_t SparseMatrixCSR<DT, MT>::getCusparseAlg() {
    return CUSPARSE_SPMM_CSR_ALG2;
}

template <typename DT, typename MT>
std::ostream &operator<<(std::ostream &out, SparseMatrixCSR<DT, MT> &m) {
    out << m.numRows << ' ' << m.numCols << ' ' << m.numNonZero << std::endl;
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

template class SparseMatrixCSR<float, uint32_t>;
template class SparseMatrixCSR<double, uint32_t>;

} // namespace cuspmm