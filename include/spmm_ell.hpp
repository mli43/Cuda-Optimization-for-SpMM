#pragma once

#include "formats/sparse_ell.hpp"

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmEllCpu(SparseMatrixELL<T>* ma, DenseMatrix<T>* mb);

template <typename T, typename AccT>
DenseMatrix<T>* spmmEllDevice(SparseMatrixELL<T>* a, DenseMatrix<T>* b);

template <typename T>
void runEngineELL(SparseMatrixELL<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol);
}