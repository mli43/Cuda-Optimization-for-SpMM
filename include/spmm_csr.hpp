#pragma once

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmCsrCpu(SparseMatrixCSR<T>* ma, DenseMatrix<T>* mb);

template <typename T, typename AccT>
DenseMatrix<T>* spmmCsrDevice(SparseMatrixCSR<T>* a, DenseMatrix<T>* b);

template <typename T>
void runEngineCSR(SparseMatrixCSR<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol);
}