#include "commons.hpp"
#include "formats/sparse_coo.hpp"

namespace cuspmm {

template <typename DT, typename MT>
SparseMatrixCOO<DT, MT>::SparseMatrixCOO() : SparseMatrix<DT, MT>() {
    this->rowIdxs = nullptr;
    this->colIdxs = nullptr;
}

template <typename DT, typename MT>
SparseMatrixCOO<DT, MT>::SparseMatrixCOO(std::string filePath) {
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

template <typename DT, typename MT>
SparseMatrixCOO<DT, MT>::SparseMatrixCOO(MT numRows,
                                    MT numCols,
                                    MT numNonZero,
                                    bool onDevice)
    : rowIdxs(nullptr), colIdxs(nullptr) {
    this->numRows = numRows;
    this->numCols = numCols;
    this->numNonZero = numNonZero;
    this->onDevice = onDevice;
    this->allocateSpace(onDevice);
}

template <typename DT, typename MT> SparseMatrixCOO<DT, MT>::~SparseMatrixCOO() {
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

template <typename DT, typename MT>
void SparseMatrixCOO<DT, MT>::setCusparseSpMatDesc(cusparseSpMatDescr_t *matDescP) {
    cudaDataType dt;
    if constexpr (std::is_same<DT, half>::value) {
        dt = CUDA_R_16F;
    } else if constexpr (std::is_same<DT, float>::value) {
        dt = CUDA_R_32F;
    } else if constexpr (std::is_same<DT, double>::value) {
        dt = CUDA_R_64F;
    }
    assertTypes3(DT, half, float, double);

    CHECK_CUSPARSE(cusparseCreateCoo(
        matDescP, this->numRows, this->numCols, this->numNonZero, this->rowIdxs,
        this->colIdxs, this->data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        dt));
}

template <typename DT, typename MT>
cusparseSpMMAlg_t SparseMatrixCOO<DT, MT>::getCusparseAlg() {
    return CUSPARSE_SPMM_COO_ALG4;
}

template <typename DT, typename MT> SparseMatrixCOO<DT, MT> *SparseMatrixCOO<DT, MT>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixCOO<DT, MT> *newMatrix = new SparseMatrixCOO<DT, MT>(
        this->numRows, this->numCols, this->numNonZero, true);

    cudaCheckError(cudaMemcpy(newMatrix->rowIdxs, this->rowIdxs,
                              this->numNonZero * sizeof(DT),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->colIdxs, this->colIdxs,
                              this->numNonZero * sizeof(DT),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                              this->numNonZero * sizeof(DT),
                              cudaMemcpyHostToDevice));
    return newMatrix;
}

template <typename DT, typename MT> bool SparseMatrixCOO<DT, MT>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    assert(this->rowIdxs == nullptr);
    assert(this->colIdxs == nullptr);
    if (onDevice) {
        cudaCheckError(cudaMalloc(&this->data, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMalloc(
            &this->rowIdxs, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMalloc(
            &this->colIdxs, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMemset(this->data, 0, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMemset(
            this->rowIdxs, 0, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMemset(
            this->colIdxs, 0, this->numNonZero * sizeof(DT)));
    } else {
        cudaCheckError(
            cudaMallocHost(&this->data, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMallocHost(
            &this->rowIdxs, this->numNonZero * sizeof(DT)));
        cudaCheckError(cudaMallocHost(
            &this->colIdxs, this->numNonZero * sizeof(DT)));
        std::memset(this->data, 0, this->numNonZero * sizeof(DT));
        std::memset(this->rowIdxs, 0,
                    this->numNonZero * sizeof(DT));
        std::memset(this->colIdxs, 0,
                    this->numNonZero * sizeof(DT));
    }

    return true;
}

template <typename DT, typename MT> DenseMatrix<DT, MT> *SparseMatrixCOO<DT, MT>::toDense() {
    assert(!this->onDevice);

    using mt = MT;

    DenseMatrix<DT, MT> *dm =
        new DenseMatrix<DT, MT>(this->numRows, this->numCols, false);

    for (size_t i = 0; i < this->numNonZero; i++) {
        mt r = this->rowIdxs[i];
        mt c = this->colIdxs[i];
        dm->data[r * dm->numCols + c] = this->data[i];
    }

    return dm;
}

template <typename DT, typename MT>
std::ostream &operator<<(std::ostream &out, SparseMatrixCOO<DT, MT> &m) {
    out << m.numRows << ' ' << m.numCols << ' ' << m.numNonZero << std::endl;
    for (size_t i = 0; i < m->numNonZero; i++) {
        std::cout << m.rowIdxs[i] << ' ' << m.colIdxs[i] << ' ' << m.data[i]
                  << std::endl;
    }

    return out;
}

template class SparseMatrixCOO<float, uint32_t>;
template class SparseMatrixCOO<double, uint32_t>;
} // namespace cuspmm