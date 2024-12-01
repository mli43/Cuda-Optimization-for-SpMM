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

def process_directory(root, files):
    # Initialize variables
    dense_matrix = None
    sparse_matrix = None
    result_path = None
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

        elif file == "result.out":
            result_path = os.path.join(root, file)

        elif file.endswith(".expect"):
            expect_file_path = os.path.join(root, file)

    # Perform multiplication and comparison
    if dense_matrix is not None and sparse_matrix is not None:
        # If .expect file exists, load and compare with result.out
        if expect_file_path:
            print(f"\tExpect file found: {expect_file_path}")
            expected_result = load_result(expect_file_path, sparse_matrix.shape[0], dense_matrix.shape[1])
            print("\tLoaded expected result from .expect file.")

            # Perform the calculation
            calculated_result = calculate_result(sparse_matrix, dense_matrix)

            if np.allclose(calculated_result, expected_result):
                print("The calculated result matches the expected result in .expect.\n")
            else:
                print("The calculated result does NOT match the expected result in .expect.\n")
                diff = np.abs(calculated_result - expected_result)
                print("Difference matrix:")
                print(diff)
                print("\n")

        # If .expect file is not found, compare with result.out
        elif result_path:
            print(f"\tResult file found: {result_path}")
            expected_result = load_result(result_path, sparse_matrix.shape[0], dense_matrix.shape[1])
            print("\tLoaded expected result from result.out file.")

            # Perform the calculation
            calculated_result = calculate_result(sparse_matrix, dense_matrix)

            if np.allclose(calculated_result, expected_result):
                print("The calculated result matches the result.out file.\n")
            else:
                print("The calculated result does NOT match the result.out file.\n")
                diff = np.abs(calculated_result - expected_result)
                print("Difference matrix:")
                print(diff)
                print("\n")

        else:
            print("No .expect or result.out file found in the directory.")
    else:
        print(f"Skipping directory {root}: missing required mtx files.\n")

def validate(directory):
    # Walk through the directory and subdirectories
    for root, _, files in os.walk(directory):
        print(f"Processing directory: {root}")
        process_directory(root, files)

if __name__ == "__main__":
    # Check if a directory argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    # Get the directory from command line arguments
    input_directory = sys.argv[1]

    # Process the directory and its subdirectories
    validate(input_directory)

