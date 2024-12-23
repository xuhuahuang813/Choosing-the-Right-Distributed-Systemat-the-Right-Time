import numpy as np
import os

def generate_positive_definite_matrices(num_matrices, matrix_size, output_dir):
    """
    Generates multiple positive definite matrices and saves them to CSV files.
    :param num_matrices: Number of matrices to generate
    :param matrix_size: Size of each matrix (matrix_size x matrix_size)
    :param output_dir: Directory to save the matrix files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_matrices):
        # Generate a random matrix A
        A = np.random.randint(1, 1000, size=(matrix_size, matrix_size))
        # Generate a positive definite matrix: A^T * A
        # matrix = np.dot(A.T, A)
        matrix_filename = os.path.join(output_dir, f"matrix_{i+1}.csv")
        np.savetxt(matrix_filename, A, delimiter=',')  # Save matrix to CSV file

# Example usage
generate_positive_definite_matrices(num_matrices=1000, matrix_size=1000, output_dir='/home/ubuntu/mapreduce/matrix')