#include "format.hpp"
#include "spmm_csr.hpp"

int main(int argc, char *argv[]) {
    std::string filePath = argv[1];
    std::string filePathDense = argv[2];

    cuspmm::SparseMatrixCSR<float> *a = new cuspmm::SparseMatrixCSR<float>(filePath);
    cuspmm::DenseMatrix<float> *b = new cuspmm::DenseMatrix<float>(filePathDense);

    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    cuspmm::runEngineCSR<float>(a, b, abs_tol, rel_tol);

    return 0;
}