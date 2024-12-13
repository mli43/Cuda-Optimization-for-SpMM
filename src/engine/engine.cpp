#include "format.hpp"

namespace cuspmm {
// template <typename MatT>
// void runEngine(MatT *a, DenseMatrix<T>* b, float abs_tol, double rel_tol) {
//     auto start = std::chrono::high_resolution_clock::now();

//     // 1. Move to device
//     SparseMatrixCOO<T>* da = a->copy2Device();
//     DenseMatrix<T>* db = b->copy2Device();
//     auto copy_to_device_end = std::chrono::high_resolution_clock::now();

//     // 2. Launch kernel
//     auto cRes = spmmCooDevice<T, double>(da, db);
//     auto kernel_end = std::chrono::high_resolution_clock::now();

//     auto cResCpu = cRes->copy2Host();
//     auto copy_to_host_end = std::chrono::high_resolution_clock::now();

//     // 3. Check result
//     auto cResSeq = spmmCooCpu<T, double>(a, b);
//     auto seq_end = std::chrono::high_resolution_clock::now();

//     // 4. Report time 
//     auto copy2DeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_device_end - start);
//     auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - copy_to_device_end);
//     auto copy2HostTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_host_end - kernel_end);
//     auto parallelTime = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_host_end - start);
//     auto seqTime = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - copy_to_host_end);

//     std::cout << "copy2DeviceTime (us):" << copy2DeviceTime.count() << ','
//               << "kernelTime (us):" << kernelTime.count() << ','
//               << "copy2HostTime (us):" << copy2HostTime.count() << ','
//               << "parallelTime (us):" << parallelTime.count() << ','
//               << "seqTime (us):" << seqTime.count() << '\n';


//     cResCpu->save2File("coo_cuda.res");
//     cResSeq->save2File("coo_cpu.res");

//     auto denseA = a->toDense();
//     auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
//     torch::Tensor taDevice = torch::from_blob(denseA->data, {denseA->numRows, denseA->numCols}, options).clone().cuda();
//     torch::Tensor tbDevice = torch::from_blob(b->data, {b->numRows, b->numCols}, options).clone().cuda();
//     torch::Tensor tcCpu = torch::from_blob(cResCpu->data, {cResCpu->numRows, cResCpu->numCols}, options).clone();
//     torch::Tensor cResTorch = torch::matmul(taDevice, tbDevice).cpu();
//     std::cout << "coo allclose: " << torch::allclose(tcCpu, cResTorch, rel_tol, abs_tol) << std::endl;

//     auto denseTorch = new DenseMatrix<T>(cResCpu->numRows, cResCpu->numCols, false);
//     std::memcpy(denseTorch->data, cResTorch.data_ptr<float>(), denseTorch->numRows * denseTorch->numCols * sizeof(float));
//     denseTorch->save2File("coo_torch.res");
// }

// template void runEngine<float>(SparseMatrix<float> *a, DenseMatrix<float>* b, float abs_tol, double rel_tol);

} // namespace cuspmm