#pragma once

namespace cuspmm {
template <typename DT, typename MT>
DenseMatrix<DT, MT>* cusparseTest(SparseMatrix<DT, MT>* a, DenseMatrix<DT, MT>* b, DenseMatrix<DT, MT>* c);
}