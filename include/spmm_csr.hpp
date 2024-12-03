namespace cuspmm {

template <typename T, typename AccT>
DenseMatrix<T>* spmmCsrCpu(SparseMatrixCSR<T>* ma, DenseMatrix<T>* mb);

template <typename T, typename AccT>
DenseMatrix<T>* spmmCsrDevice(SparseMatrixCSR<T>* a, DenseMatrix<T>* b);

}