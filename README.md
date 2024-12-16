# Cuda-Optimization-for-SpMM
This is the project repo for CMU 15618 final project

## Reports
[Proposal](reports/proposal.pdf)

[Milestone](reports/milestone.pdf)

[Final](reports/final.pdf)

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
optimization with the NVIDIA cuSparse library to understand and evaluate the effectiveness of our implementation.

## How to compile

This repository requires installing 


> CUDA>=12.4

> torch C++ library (Please refer to [Installing C++ Distributions of PyTorch](https://pytorch.org/cppdocs/installing.html#))

> CMAKE>=3.20

Replace your torch path with $/15618/libtorch$ in CMakeLists.txt. Run the following commands:

```
cmake -B build
cd build
make -B -j4
```

## How to run

1. Compile the program
2. Run `bash ./scripts/data.sh` to automatically generate necessary data files.
3. Run `cd build && ./cuspmm --csr --coo --bsr --ell -d <data_directory>`. You can refer `-h` option for help message.