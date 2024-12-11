#pragma once

#include "formats/sparse_bsr.hpp"

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmBsrCpu(SparseMatrixBSR<T>* ma, DenseMatrix<T>* mb);

template <typename T, typename AccT>
DenseMatrix<T>* spmmBsrDevice(SparseMatrixBSR<T>* a, DenseMatrix<T>* b);

template <typename T>
void runEngineBSR(SparseMatrixBSR<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol);
}