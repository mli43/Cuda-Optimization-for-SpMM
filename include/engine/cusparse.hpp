#pragma once

namespace cuspmm {
template <typename DataT>
DenseMatrix<DataT>* cusparseTest(SparseMatrix<DataT>* a, DenseMatrix<DataT>* b);
}