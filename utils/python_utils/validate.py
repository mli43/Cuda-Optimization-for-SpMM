import os
import sys
from scipy.io import mmread
import numpy as np

def load_result(file_path, num_rows, num_cols):
    """Load the result.out file without size information, assuming row-major format."""
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            result.append(list(map(float, line.split())))
    result = np.array(result)
    # Ensure the result has the correct dimensions
    if result.shape != (num_rows, num_cols):
        raise ValueError(f"Expected result dimensions ({num_rows}, {num_cols}), but got {result.shape}")
    return result

def calculate_result(sparse_matrix, dense_matrix):
    """Calculate the result of sparse @ dense multiplication."""
    return sparse_matrix @ dense_matrix

def save_result_to_file(root, result_matrix):
    result_file_path = os.path.join(root, "result.expect")
    rows, cols = result_matrix.shape
    with open(result_file_path, 'w') as f:
        for row in range(rows):
            # f.write(" ".join(map(str, result_matrix[row])))
            f.write(" ".join(f"{value:.10f}" for value in result_matrix[row]) + "\n")
    return result_file_path

def process_directory(root, files):
    # Initialize variables
    dense_matrix = None
    sparse_matrix = None
    result_path = None
    out_files = []
    expect_file_path = None

    # Traverse the files in the current directory
    for file in files:
        if file == "dense.mtx":
            dense_mtx_path = os.path.join(root, file)
            # Read the dense.mtx file
            try:
                matrix = mmread(dense_mtx_path)
                dense_matrix = matrix.todense()
            except Exception as e:
                print(f"Error reading {dense_mtx_path}: {e}")
                continue

        elif file.endswith(".mtx") and file != "dense.mtx":
            mtx_file_path = os.path.join(root, file)
            # Convert the matrix to COO format
            try:
                sparse_matrix = mmread(mtx_file_path).tocoo()
            except Exception as e:
                print(f"Error reading or converting {mtx_file_path}: {e}")
                continue

        # elif file == "result.out":
        elif file.endswith(".out"):
            result_path = os.path.join(root, file)
            out_files.append(result_path)

        elif file.endswith(".expect"):
            expect_file_path = os.path.join(root, file)

    # Perform multiplication and comparison
    if (dense_matrix is None or sparse_matrix is None):
        print(f"Skipping directory {root}: missing required mtx files.\n")
        return

    elif expect_file_path:
        # .expect file exists, load and compare with result.out
        print(f"\tExpect file found: {expect_file_path}")
        expected_result = load_result(expect_file_path, sparse_matrix.shape[0], dense_matrix.shape[1])
        print("\tLoaded expected result from .expect file.")
    else:
        # .expect not found, calculate and compare with result.out
        expected_result = calculate_result(sparse_matrix, dense_matrix).A
        print(f"\tCalculated expected result from input files {mtx_file_path} @ {dense_mtx_path}")

        expected_path = save_result_to_file(root, expected_result)
        print(f"\tSaved expected result to {expected_path}\n")

    for result_file in out_files:
        result = load_result(result_file, sparse_matrix.shape[0], dense_matrix.shape[1])

        if np.allclose(result, expected_result):
            print(f"\tResult file {result_file} matches the expected result in .expect.")
        else:
            print(f"Result file {result_file} does NOT match the expected result in .expect.")
            diff = np.abs(result - expected_result)
            print("Difference matrix:")
            print(diff)
            print("\n")


def validate(directory):
    for root, _, files in os.walk(directory):
        print(f"Processing directory: {root}")
        process_directory(root, files)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    input_directory = sys.argv[1]

    validate(input_directory)

