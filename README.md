# Cuda-Optimization-for-SpMM
This is the project repo for CMU 15618 final project

## Reports
[Proposal](reports/proposal.pdf)


## Abstract
The ever-increasing size of deep learning models, especially LLMs, demand
higher and higher computational efficiency. Many techniques exploiting the
sparsity of model weights are invented to accelerate the inference and training
process, an example of which is pruning. After removing the less significant
weights, sparse matrix-matrix multiplication can be applied to replace dense MM
and achieve more speedup. This project focuses on implementing and optimizing
Sparse MatrixMatrix Multiplication, a.k.a. SPMM, in the context of LLM
inference. We plan to analyze the characteristics of different storage formats,
explore different algorithms, and implement a high-performance SPMM CUDA kernel
optimized with respect to GPU hardware features. We aim to reduce memory
communication overhead and increase computation speed for large SPMM in
real-world applications. We will also compare the performance of our
optimization with the NVIDIA cuSparse library and the dense counterpart
cuBLAS to understand and evaluate the effectiveness of our implementation.

