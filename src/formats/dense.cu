#include "commons.hpp"
#include "formats/dense.hpp"
#include "formats/matrix.hpp"

namespace cuspmm {
template <typename DT, typename MT>
DenseMatrix<DT, MT>::DenseMatrix(std::string filePath) {
    // Files contains only row-major contents
    this->ordering = ORDERING::ROW_MAJOR;
    this->data = nullptr;
    this->onDevice = false;

    std::ifstream inputFile(filePath);
    std::string line;

    if (!inputFile.is_open()) {
        std::cerr << "File " << filePath << "doesn't exist!" << std::endl;
        throw std::runtime_error(NULL);
    }

    inputFile >> this->numRows >> this->numCols;
    std::getline(inputFile, line); // Discard the header line

    this->allocateSpace(this->onDevice);

    for (int i = 0; i < this->numRows; i++) {
        std::getline(inputFile, line);
        std::istringstream iss(line);
        for (int j = 0; j < this->numCols; j++) {
            iss>> this->data[i * this->numCols + j];
        }
    }

}

template <typename DT, typename MT>
DenseMatrix<DT, MT>::DenseMatrix(MT numRows,
                    MT numCols, bool onDevice, ORDERING ordering) {
    this->numRows = numRows;
    this->numCols = numCols;
    this->onDevice = onDevice;
    this->ordering = ordering;
    this->data = nullptr;
    this->allocateSpace(onDevice);
}

template <typename DT, typename MT>
DenseMatrix<DT, MT>::DenseMatrix(DenseMatrix<DT, MT>* source, bool onDevice) {
    this->numRows = source->numRows;
    this->numCols = source->numCols;
    this->onDevice = onDevice;
    this->ordering = source->ordering;
    this->data = nullptr;
    this->allocateSpace(this->onDevice);
    this->copyData(source);
}

template <typename DT, typename MT>
DenseMatrix<DT, MT>::~DenseMatrix() {
    this->freeSpace();
}

template <typename DT, typename MT>
void DenseMatrix<DT, MT>::setCusparseDnMatDesc(cusparseDnMatDescr_t* matDescP) {
    cusparseOrder_t co;
    if (this->ordering == ORDERING::ROW_MAJOR) {
        co = CUSPARSE_ORDER_ROW;
    } else {
        co = CUSPARSE_ORDER_COL;
    }
    if constexpr (std::is_same<DT, half>::value) {
        CHECK_CUSPARSE(cusparseCreateDnMat(matDescP, this->numRows, this->numCols, this->numCols,
                        this->data, CUDA_R_16F, co));
    } 
    if constexpr (std::is_same<DT, float>::value) {
        CHECK_CUSPARSE(cusparseCreateDnMat(matDescP, this->numRows, this->numCols, this->numCols,
                        this->data, CUDA_R_32F, co));
    }
    if constexpr (std::is_same<DT, double>::value) {
        CHECK_CUSPARSE(cusparseCreateDnMat(matDescP, this->numRows, this->numCols, this->numCols,
                        this->data, CUDA_R_64F, co));
    } 
    assertTypes3(DT, half, float, double);
}

template <typename DT, typename MT>
bool DenseMatrix<DT, MT>::copyData(DenseMatrix<DT, MT>* source) {
    this->assertSameShape(source);
    cudaMemcpyKind type;
    if (source->onDevice && this->onDevice) {
        type = cudaMemcpyDeviceToDevice;
    } else if (source->onDevice && !this->onDevice) {
        type = cudaMemcpyDeviceToHost;
    } else if (!source->onDevice && this->onDevice) {
        type = cudaMemcpyHostToDevice;
    } else {
        type = cudaMemcpyHostToHost;
    }

    cudaCheckError(
        cudaMemcpy(this->data, source->data,
                    (this->numRows * this->numCols) * sizeof(DT),
                    type));
    
    return true;
}

template <typename DT, typename MT>
void DenseMatrix<DT, MT>::assertSameShape(DenseMatrix<DT, MT>* target) {
    assert(
        this->numRows == target->numRows &&
        this->numCols == target->numCols
    );
}

template <typename DT, typename MT>
DenseMatrix<DT, MT>* DenseMatrix<DT, MT>::copy2Device() {
    assert(this->onDevice == false);
    assert(this->data != nullptr);

    DenseMatrix<DT, MT>* newMatrix = new DenseMatrix<DT, MT>(this, true);
    return newMatrix;
}

template <typename DT, typename MT>
DenseMatrix<DT, MT>* DenseMatrix<DT, MT>::copy2Host() {
    assert(this->onDevice == true);
    assert(this->data != nullptr);

    DenseMatrix<DT, MT>* newMatrix = new DenseMatrix<DT, MT>(this, false);
    return newMatrix;
}

template <typename DT, typename MT>
bool DenseMatrix<DT, MT>::toOrdering(ORDERING newOrdering) {
    if (this->ordering == newOrdering) {
        return true;
    }

    // Malloc new space
    size_t totalSize = this->numRows * this->numCols * sizeof(DT);
    DT* newData;
    cudaCheckError(cudaMallocHost(&newData, totalSize));

    // If on device, copy to host
    if (this->onDevice) {
        cudaCheckError(cudaMemcpy(newData, this->data, totalSize, cudaMemcpyDeviceToHost));
    }

    if (this->ordering == ORDERING::ROW_MAJOR && newOrdering == ORDERING::COL_MAJOR) {
        // Reorganize
        for (int r = 0; r < this->numRows; r++) {
            for (int c = 0; c < this->numCols; c++) {
                newData[ColMjIdx(r, c, this->numRows)] = this->data[RowMjIdx(r, c, this->numCols)];
            }
        }
    } else if (this->ordering == ORDERING::COL_MAJOR && newOrdering == ORDERING::ROW_MAJOR) {
        // Reorganize
        for (int r = 0; r < this->numRows; r++) {
            for (int c = 0; c < this->numCols; c++) {
                newData[RowMjIdx(r, c, this->numCols)] = this->data[ColMjIdx(r, c, this->numRows)];
            }
        }
    } else {
        throw std::runtime_error("Incorrect ordering value");
        return false;
    }

    this->freeSpace();
    this->allocateSpace(this->onDevice);
    if (this->onDevice) {
        cudaCheckError(cudaMemcpy(this->data, newData, totalSize, cudaMemcpyHostToDevice));
    } else {
        this->data = newData;
    }

    this->ordering = newOrdering;

    return true;
}

template <typename DT, typename MT>
bool DenseMatrix<DT, MT>::save2File(std::string filePath) {
    using mt = MT;
    assert(!this->onDevice);
    assert(this->ordering == ORDERING::ROW_MAJOR);

    std::ofstream outputFile(filePath);
    if (!outputFile.is_open()) {
        std::cerr << "Cannot open output file " << filePath << std::endl;
        return false;
    }

    outputFile << this->numRows << ' ' << this->numCols << std::endl;
    for (mt r = 0; r < this->numRows; r++) {
        for (mt c = 0; c < this->numCols; c++) {
            outputFile << this->data[r * this->numCols + c] << ' ';
        }
        outputFile << std::endl;
    }

    return true;
}

template <typename DT, typename MT>
bool DenseMatrix<DT, MT>::allocateSpace(bool onDevice) {
    assert(this->data == nullptr);

    size_t totalSize = this->numRows * this->numCols * sizeof(DT);
    if (onDevice) {
        cudaCheckError(cudaMalloc(
            &this->data, totalSize));
        cudaCheckError(cudaMemset(this->data, 0, totalSize));
    } else {
        cudaCheckError(cudaMallocHost(
            &this->data, totalSize));
        std::memset(this->data, 0, totalSize);
    }
    this->onDevice = onDevice;

    return true;
}

template <typename DT, typename MT>
bool DenseMatrix<DT, MT>::freeSpace() {
    if (this->onDevice) {
        cudaCheckError(cudaFree(this->data));
    } else {
        cudaCheckError(cudaFreeHost(this->data));
    }
    this->data = nullptr;
    return true;
}

template class DenseMatrix<float, uint32_t>;
template class DenseMatrix<double, uint32_t>;
}