import os
import sys
import numpy as np
from scipy.sparse import random, csr_matrix

# Get the absolute directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the script directory to sys.path
sys.path.append(script_dir)

from convert_matrix import save_bsr_matrix, save_coo_matrix, save_csr_matrix, save_dense_matrix

def generate_sparse_matrix(rows, cols, sparsity, data_range):
    """
    Generate a sparse matrix with customizable size, sparsity, and data range.

    Parameters:
        rows (int): Number of rows in the sparse matrix.
        cols (int): Number of columns in the sparse matrix.
        sparsity (float): Fraction of non-zero elements (0.0 to 1.0).
        data_range (tuple): Range of data values as (min, max).

    Returns:
        csr_matrix: Generated sparse matrix in CSR format.
    """
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be between 0 and 1.")

    min_val, max_val = data_range
    if min_val >= max_val:
        raise ValueError("Minimum value of data range must be less than the maximum.")

    # Generate random sparse matrix
    sparse_matrix = random(rows, cols, density=sparsity, format='csr',
                           data_rvs=lambda s: np.random.uniform(min_val, max_val, size=s).astype(np.float32))

    return sparse_matrix


def generate_dense_matrix(rows, cols, data_range):
    """
    Generate a dense matrix with customizable size and data range.

    Parameters:
        rows (int): Number of rows in the dense matrix.
        cols (int): Number of columns in the dense matrix.
        data_range (tuple): Range of data values as (min, max).

    Returns:
        np.ndarray: Generated dense matrix.
    """
    min_val, max_val = data_range
    if min_val >= max_val:
        raise ValueError("Minimum value of data range must be less than the maximum.")

    # Generate dense matrix
    dense_matrix = np.random.uniform(min_val, max_val, size=(rows, cols)).astype(np.float32)

    return dense_matrix


if __name__ == "__main__":
    # Example usage
    data_range = (1, 10) # Data range (min, max)

    data_dir = sys.argv[1]

    rows = 2048
    cols = 2048
    sparsity_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for sp in sparsity_list:
        dir_name = f"sp_{sp}_2048x2048"
        mt_dir = os.path.join(data_dir, dir_name)
        os.makedirs(mt_dir, True)
        sparse_matrix = generate_sparse_matrix(rows, cols, sp, (-100, 100))
        save_csr_matrix(sparse_matrix.tocsr(), os.path.join(mt_dir, "matrix.csr"))
        save_coo_matrix(sparse_matrix.tocoo(), os.path.join(mt_dir, "matrix.coo"))

        dense = generate_dense_matrix(cols, 1024, (-100, 100))
        save_dense_matrix(dense, os.path.join(mt_dir, "dense.in"))

        print("complete sparsity ", sp)
