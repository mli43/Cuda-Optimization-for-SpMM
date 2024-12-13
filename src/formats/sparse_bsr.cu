#include "commons.hpp"
#include "formats/sparse_bsr.hpp"

namespace cuspmm {
template <typename DT, typename MT>
SparseMatrixBSR<DT, MT>::SparseMatrixBSR() : SparseMatrix<DT, MT>() {
    this->blockRowSize = 0;
    this->blockColSize = 0;
    this->numBlocks = 0;
    this->numBlockRows = 0;
    this->numBlockRows = 0;
    this->numElements = 0;
    this->blockRowPtrs = nullptr;
    this->blockColIdxs = nullptr;
}

template <typename DT, typename MT>
SparseMatrixBSR<DT, MT>::SparseMatrixBSR(std::string filePath) {
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

template <typename DT, typename MT>
SparseMatrixBSR<DT, MT>::SparseMatrixBSR(MT numRows,
                                    MT numCols,
                                    MT numNonZero,
                                    MT blockRowSize,
                                    MT blockColSize,
                                    MT numBlocks,
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

template <typename DT, typename MT>
SparseMatrixBSR<DT, MT>::SparseMatrixBSR(SparseMatrixBSR<DT, MT> *target, bool onDevice) {
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

template <typename DT, typename MT> SparseMatrixBSR<DT, MT>::~SparseMatrixBSR() {
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

template <typename DT, typename MT>
void SparseMatrixBSR<DT, MT>::setCusparseSpMatDesc(cusparseSpMatDescr_t *matDescP) {
    cudaDataType dt;
    if constexpr (std::is_same<DT, half>::value) {
        dt = CUDA_R_16F;
    } else if constexpr (std::is_same<DT, float>::value) {
        dt = CUDA_R_32F;
    } else if constexpr (std::is_same<DT, double>::value) {
        dt = CUDA_R_64F;
    }
    assertTypes3(DT, half, float, double);

    CHECK_CUSPARSE(cusparseCreateBsr(
        matDescP, this->numBlockRows, (this->numCols / this->blockColSize),
        this->numBlocks, this->blockRowSize, this->blockColSize,
        this->blockRowPtrs, this->blockColIdxs, this->data, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dt, CUSPARSE_ORDER_ROW));
}

template <typename DT, typename MT>
cusparseSpMMAlg_t SparseMatrixBSR<DT, MT>::getCusparseAlg() {
    return CUSPARSE_SPMM_ALG_DEFAULT;
}

template <typename DT, typename MT>
bool SparseMatrixBSR<DT, MT>::copyData(SparseMatrixBSR<DT, MT> *source, bool onDevice) {
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
        (this->numBlockRows + 1) * sizeof(DT), type));
    cudaCheckError(cudaMemcpy(this->blockColIdxs, source->blockColIdxs,
                              this->numBlocks * sizeof(DT),
                              type));
    cudaCheckError(cudaMemcpy(this->data, source->data,
                              this->numElements * sizeof(DT), type));

    return true;
}

template <typename DT, typename MT> SparseMatrixBSR<DT, MT> *SparseMatrixBSR<DT, MT>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    SparseMatrixBSR<DT, MT> *newMatrix = new SparseMatrixBSR<DT, MT>(this, true);

    return newMatrix;
}

template <typename DT, typename MT> void SparseMatrixBSR<DT, MT>::assertCheck() {
    assert(this->numRows % this->blockRowSize == 0);
    assert(this->numCols % this->blockColSize == 0);
}

template <typename DT, typename MT>
void SparseMatrixBSR<DT, MT>::assertSameShape(SparseMatrixBSR<DT, MT> *target) {
    assert(this->blockRowSize == target->blockRowSize &&
           this->blockColSize == target->blockColSize &&
           this->numBlocks == target->numBlocks &&
           this->numBlockRows == target->numBlockRows &&
           this->numRows == target->numRows &&
           this->numCols == target->numCols &&
           this->numNonZero == target->numNonZero);
}

template <typename DT, typename MT> bool SparseMatrixBSR<DT, MT>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);
    if (onDevice) {
        cudaCheckError(cudaMalloc(&this->data, this->numElements * sizeof(DT)));
        cudaCheckError(
            cudaMemset(this->data, 0, this->numElements * sizeof(DT)));

        cudaCheckError(
            cudaMalloc(&this->blockRowPtrs, (this->numBlockRows + 1) *
                                                sizeof(DT)));
        cudaCheckError(cudaMemset(this->blockRowPtrs, 0,
                                  (this->numBlockRows + 1) *
                                      sizeof(DT)));

        cudaCheckError(
            cudaMalloc(&this->blockColIdxs,
                       this->numBlocks * sizeof(DT)));
        cudaCheckError(
            cudaMemset(this->blockColIdxs, 0,
                       this->numBlocks * sizeof(DT)));
    } else {
        cudaCheckError(
            cudaMallocHost(&this->data, this->numElements * sizeof(DT)));
        std::memset(this->data, 0, this->numElements * sizeof(DT));

        cudaCheckError(cudaMallocHost(&this->blockRowPtrs,
                                      (this->numBlockRows + 1) *
                                          sizeof(DT)));
        std::memset(this->blockRowPtrs, 0,
                    (this->numBlockRows + 1) * sizeof(DT));

        cudaCheckError(
            cudaMallocHost(&this->blockColIdxs,
                           this->numBlocks * sizeof(DT)));
        std::memset(this->blockColIdxs, 0,
                    this->numBlocks * sizeof(DT));
    }

    return true;
}

template <typename DT, typename MT>
SparseMatrixBSR<DT, MT> *
SparseMatrixBSR<DT, MT>::fromDense(DenseMatrix<DT, MT> *dense,
                              MT blockRowSize,
                              MT blockColSize) {
    throw std::runtime_error("Not Implemented");
    const DT zero = 0;
    assert(dense->numRows % blockRowSize == 0 &&
           dense->numCols % blockColSize == 0);

    using mt = MT;
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

template <typename DT, typename MT> DenseMatrix<DT, MT> *SparseMatrixBSR<DT, MT>::toDense() {
    assert(!this->onDevice);

    using mt = MT;

    DenseMatrix<DT, MT> *dm =
        new DenseMatrix<DT, MT>(this->numRows, this->numCols, false);

    for (mt blockRow = 0; blockRow < this->numBlockRows; blockRow++) {
        mt blockRowStart = this->blockRowPtrs[blockRow];
        mt blockRowEnd = this->blockRowPtrs[blockRow + 1];
        for (mt blockIdx = blockRowStart; blockIdx < blockRowEnd; blockIdx++) {
            mt blockCol = this->blockColIdxs[blockIdx];
            DT *blockData = this->data +
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

template <typename DT, typename MT>
std::ostream &operator<<(std::ostream &out, SparseMatrixBSR<DT, MT> &m) {
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

template class SparseMatrixBSR<float, uint32_t>;
template class SparseMatrixBSR<double, uint32_t>;
} // namespace cuspmm