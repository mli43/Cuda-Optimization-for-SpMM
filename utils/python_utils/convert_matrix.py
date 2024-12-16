import os
import sys
from scipy.io import mmread
from scipy.sparse import random, csr_matrix, coo_matrix, bsr_matrix
import numpy as np

def save_bsr_matrix(matrix, filename, block_size=(4, 4)):
    """
    Save a bsr_matrix to a file in the specified custom format.

    Parameters:
        matrix (bsr_matrix): The BSR matrix to save.
        filename (str): Path to the file where the matrix will be saved.
    """
    if not isinstance(matrix, bsr_matrix):
        raise ValueError("Input matrix must be a bsr_matrix.")
    
    rows, cols = matrix.shape
    size = block_size[0]
    while rows % size != 0 and cols % size != 0:
        size = size // 2
    print(f"bsr using shape {size},{size}")
    try :
        matrix = matrix.tobsr((size, size), copy=True)
    except:
        try:
            matrix = matrix.tobsr((1, 1), copy=True)
        except:
            raise RuntimeError("wrong")

    # Extract BSR matrix data
    num_rows, num_cols = matrix.shape
    block_size = matrix.blocksize
    block_row_size, block_col_size = block_size
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr

    # Number of non-zero elements
    nnz = data.size

    # Number of blocks
    num_blocks = len(indices)

    with open(filename, 'w') as file:
        # Write the header information
        file.write(f"{num_rows} {num_cols} {nnz} {block_row_size} {block_col_size} {num_blocks}\n")

        # Write block offsets (indptr)
        file.write(" ".join(map(str, indptr)) + "\n")

        # Write block column indices
        file.write(" ".join(map(str, indices)) + "\n")

        # Write the values of the blocks
        for block in data:
            # print(block)
            file.write(" ".join(map(str, block.flatten())) + "\n")

    print(f"Saved BSR matrix to {filename}")

        
def save_csr_matrix(csr_matrix, file_path):
    """
    Save a CSR matrix to a file in the specified format.

    Parameters:
        csr_matrix (csr_matrix): The CSR matrix to save.
        file_path (str): Path to the output file.

    Format:
        num_rows num_cols num_non_zero
        row_ptrs
        column_indices
        values
    """
    num_rows, num_cols = csr_matrix.shape
    num_non_zero = csr_matrix.nnz
    row_ptrs = csr_matrix.indptr
    column_indices = csr_matrix.indices
    values = csr_matrix.data

    with open(file_path, 'w') as f:
        # Write header
        f.write(f"{num_rows} {num_cols} {num_non_zero}\n")
        # Write row pointers
        f.write(" ".join(map(str, row_ptrs)) + "\n")
        # Write column indices
        f.write(" ".join(map(str, column_indices)) + "\n")
        # Write values
        f.write(" ".join(map(str, values)) + "\n")

    print(f"Saved csr format to {file_path}")



def save_coo_matrix(matrix_coo, filepath):

    with open(filepath, "w") as out_file:
        # Line 1: Number of rows, columns, and non-zero elements
        rows, cols = matrix_coo.shape
        nnz = matrix_coo.nnz
        out_file.write(f"{rows} {cols} {nnz}\n")
        
        rows, cols, values = matrix_coo.row, matrix_coo.col, matrix_coo.data

        # Sort by row, then by column (row-major order)
        sorted_indices = np.lexsort((cols, rows))  # Sort by rows, then by cols
        rows = rows[sorted_indices]
        cols = cols[sorted_indices]
        values = values[sorted_indices]
        
        for r, c, v in zip(rows, cols, values):
            out_file.write(f"{r} {c} {v}\n")

    print(f"Saved COO format to {filepath}")

def save_dense_matrix(dense_matrix, file_path):
    """
    Save a dense matrix to a file in the specified format.

    Parameters:
        dense_matrix (np.ndarray): The dense matrix to save.
        file_path (str): Path to the output file.

    Format:
        num_rows num_cols
        data_row_1 ...
        ...
        data_row_n ...
    """
    num_rows, num_cols = dense_matrix.shape

    with open(file_path, 'w') as f:
        # Write header
        f.write(f"{num_rows} {num_cols}\n")
        # Write data rows
        for row in dense_matrix:
            f.write(" ".join(map(str, row)) + "\n")