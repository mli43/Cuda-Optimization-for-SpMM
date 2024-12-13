#include "commons.hpp"
#include "formats/sparse_ell.hpp"

namespace cuspmm {

template <typename DT, typename MT>
SparseMatrixELL<DT, MT>::SparseMatrixELL() : SparseMatrix<DT, MT>() {
    this->colIdxs = nullptr;
    this->maxRowNnz = 0;
}

template <typename DT, typename MT>
SparseMatrixELL<DT, MT>::SparseMatrixELL(std::string colindPath,
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

template <typename DT, typename MT>
SparseMatrixELL<DT, MT>::SparseMatrixELL(MT numRows,
                                    MT numCols,
                                    MT numNonZero,
                                    MT maxRowNnz,
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

template <typename DT, typename MT> SparseMatrixELL<DT, MT>::~SparseMatrixELL() {
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
void SparseMatrixELL<DT, MT>::setCusparseSpMatDesc(cusparseSpMatDescr_t *matDescP) {
    cudaDataType dt;
    if constexpr (std::is_same<DT, half>::value) {
        dt = CUDA_R_16F;
    } else if constexpr (std::is_same<DT, float>::value) {
        dt = CUDA_R_32F;
    } else if constexpr (std::is_same<DT, double>::value) {
        dt = CUDA_R_64F;
    }
    assertTypes3(DT, half, float, double);

    // FIXME:
    throw std::runtime_error("not implemented");
}

template <typename DT, typename MT>
cusparseSpMMAlg_t SparseMatrixELL<DT, MT>::getCusparseAlg() {
    return CUSPARSE_SPMM_ALG_DEFAULT;
}

template <typename DT, typename MT> bool SparseMatrixELL<DT, MT>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    assert(this->colIdxs == nullptr);
    if (onDevice) {

        cudaCheckError(cudaMalloc(&this->data,
                                  this->numRows * this->maxRowNnz * sizeof(DT)));
        cudaCheckError(
            cudaMalloc(&this->colIdxs, this->numRows * this->maxRowNnz *
                                           sizeof(DT)));
        cudaCheckError(cudaMemset(this->data, 0,
                                  this->numRows * this->maxRowNnz * sizeof(DT)));
        cudaCheckError(cudaMemset(this->colIdxs, 0,
                                  this->numRows * this->maxRowNnz *
                                      sizeof(DT)));
    } else {
        cudaCheckError(cudaMallocHost(
            &this->data, this->numRows * this->maxRowNnz * sizeof(DT)));
        cudaCheckError(
            cudaMallocHost(&this->colIdxs, this->numRows * this->maxRowNnz *
                                               sizeof(DT)));
        std::memset(this->data, 0, this->numRows * this->maxRowNnz * sizeof(DT));
        std::memset(this->colIdxs, 0,
                    this->numRows * this->maxRowNnz *
                        sizeof(DT));
    }

    return true;
}

template <typename DT, typename MT> SparseMatrixELL<DT, MT> *SparseMatrixELL<DT, MT>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixELL<DT, MT> *newMatrix = new SparseMatrixELL<DT, MT>(
        this->numRows, this->numCols, this->numNonZero, this->maxRowNnz, true);

    cudaCheckError(cudaMemcpy(newMatrix->colIdxs, this->colIdxs,
                              this->numRows * this->maxRowNnz *
                                  sizeof(DT),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                              this->numRows * this->maxRowNnz * sizeof(DT),
                              cudaMemcpyHostToDevice));

    return newMatrix;
}

template <typename DT, typename MT> DenseMatrix<DT, MT> *SparseMatrixELL<DT, MT>::toDense() {
    assert(!this->onDevice);

    using mt = DT;

    DenseMatrix<DT, MT> *dm =
        new DenseMatrix<DT, MT>(this->numRows, this->numCols, false);

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

template class SparseMatrixELL<float, uint32_t>;
template class SparseMatrixELL<double, uint32_t>;

} // namespace cuspmm