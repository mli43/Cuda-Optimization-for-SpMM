#include "commons.hpp"
#include "formats/sparse_ell.hpp"

namespace cuspmm {

template <typename DT, typename MT>
SparseMatrixELL<DT, MT>::SparseMatrixELL() : SparseMatrix<DT, MT>() {
    this->rowIdxs = nullptr;
    this->maxColNnz = 0;
}

template <typename DT, typename MT>
SparseMatrixELL<DT, MT>::SparseMatrixELL(std::string rowindPath,
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


template <typename DT, typename MT>
SparseMatrixELL<DT, MT>::SparseMatrixELL(MT numRows,
                                    MT numCols,
                                    MT numNonZero,
                                    MT maxColNnz,
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

template <typename DT, typename MT> SparseMatrixELL<DT, MT>::~SparseMatrixELL() {
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
    assert(this->rowIdxs == nullptr);
    if (onDevice) {

        cudaCheckError(cudaMalloc(&this->data,
                                  this->numCols * this->maxColNnz * sizeof(DT)));
        cudaCheckError(
            cudaMalloc(&this->rowIdxs, this->numCols * this->maxColNnz *
                                           sizeof(MT)));
        cudaCheckError(cudaMemset(this->data, 0,
                                  this->numCols * this->maxColNnz * sizeof(DT)));
        cudaCheckError(cudaMemset(this->rowIdxs, 0,
                                  this->numCols * this->maxColNnz *
                                      sizeof(MT)));
    } else {
        cudaCheckError(cudaMallocHost(
            &this->data, this->numCols * this->maxColNnz * sizeof(DT)));
        cudaCheckError(
            cudaMallocHost(&this->rowIdxs, this->numCols * this->maxColNnz *
                                               sizeof(MT)));
        std::memset(this->data, 0, this->numCols * this->maxColNnz * sizeof(DT));
        std::memset(this->rowIdxs, 0,
                    this->numCols * this->maxColNnz *
                        sizeof(MT));
    }

    return true;
}

template <typename DT, typename MT> SparseMatrixELL<DT, MT> *SparseMatrixELL<DT, MT>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixELL<DT, MT> *newMatrix = new SparseMatrixELL<DT, MT>(
        this->numRows, this->numCols, this->numNonZero, this->maxColNnz, true);

    cudaCheckError(cudaMemcpy(newMatrix->rowIdxs, this->rowIdxs,
                              this->numCols* this->maxColNnz *
                                  sizeof(MT),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(newMatrix->data, this->data,
                              this->numCols * this->maxColNnz * sizeof(DT),
                              cudaMemcpyHostToDevice));

    return newMatrix;
}

template <typename DT, typename MT> DenseMatrix<DT, MT> *SparseMatrixELL<DT, MT>::toDense() {
    assert(!this->onDevice);

    DenseMatrix<DT, MT> *dm =
        new DenseMatrix<DT, MT>(this->numRows, this->numCols, false);

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

template class SparseMatrixELL<float, uint32_t>;
template class SparseMatrixELL<double, uint32_t>;

} // namespace cuspmm
