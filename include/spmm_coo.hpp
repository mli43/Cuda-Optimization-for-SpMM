#pragma once

#include "formats/sparse_coo.hpp"

namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmCooCpu(SparseMatrixCOO<T>* ma, DenseMatrix<T>* mb);

template <typename T, typename AccT>
DenseMatrix<T>* spmmCooDevice(SparseMatrixCOO<T>* a, DenseMatrix<T>* b);

template <typename T>
void runEngineCOO(SparseMatrixCOO<T> *a, DenseMatrix<T>* b, float abs_tol, double rel_tol);
}