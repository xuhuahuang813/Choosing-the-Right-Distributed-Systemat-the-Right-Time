import sys
import os
import numpy as np
import time
import ray
import multiprocessing
import logging


# 配置日志记录器
logging.basicConfig(filename='experiment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 从文件读取矩阵的函数保持不变
def read_matrix_from_file(file_path):
    return np.genfromtxt(file_path, delimiter=',')


@ray.remote
def process_matrix_file_all_remote(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

@ray.remote
def process_matrix_file_power_method_remote(matrix, convergence=False, max_iterations=100, tolerance=1e-6):
    vector = np.random.rand(matrix.shape[1])
    prev_eigenvalue = None
    for i in range(max_iterations):
        new_vector = np.dot(matrix, vector)
        new_vector_norm = np.linalg.norm(new_vector)
        new_vector = new_vector / new_vector_norm
        eigenvalue = np.dot(new_vector, np.dot(matrix, new_vector)) / np.dot(new_vector, new_vector)
        if convergence and prev_eigenvalue is not None and np.abs(eigenvalue - prev_eigenvalue) < tolerance:
            break
        prev_eigenvalue = eigenvalue
        vector = new_vector
    return eigenvalue, new_vector

def process_matrix_file_all(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def process_matrix_file_power_method(matrix, convergence=False, max_iterations=100, tolerance=1e-6):
    vector = np.random.rand(matrix.shape[1])
    prev_eigenvalue = None
    for i in range(max_iterations):
        new_vector = np.dot(matrix, vector)
        new_vector_norm = np.linalg.norm(new_vector)
        new_vector = new_vector / new_vector_norm
        eigenvalue = np.dot(new_vector, np.dot(matrix, new_vector)) / np.dot(new_vector, new_vector)
        if convergence and prev_eigenvalue is not None and np.abs(eigenvalue - prev_eigenvalue) < tolerance:
            break
        prev_eigenvalue = eigenvalue
        vector = new_vector
    return eigenvalue, new_vector

def run_experiment(matrix_dir, file_counts, mode='ray', method='power_method', convergence=False):
    ray.init()
    matrix_files = [os.path.join(matrix_dir, f) for f in os.listdir(matrix_dir) if f.endswith('.csv')]
    for count in file_counts:
        # logging.info(f"Running experiment for the first {count} files")
        matrix_files_to_process = matrix_files[:count]
        matrices = [read_matrix_from_file(matrix_file) for matrix_file in matrix_files_to_process]
        start_time = time.time()
        if mode == 'serial':
            for i in range(count):
                if method == 'power_method':
                    process_matrix_file_power_method(matrices[i], convergence)
                else:
                    process_matrix_file_all(matrices[i])
        elif mode == 'multiprocessing':
            if method == 'power_method':
                with multiprocessing.Pool() as pool:
                    processing_tasks = [(matrix, convergence) for matrix in matrices]
                    results = pool.starmap(process_matrix_file_power_method, processing_tasks)
            else:
                with multiprocessing.Pool() as pool:
                    results = pool.map(process_matrix_file_all, matrices)
        elif mode == 'ray':
            if method == 'power_method':
                processing_tasks = [process_matrix_file_power_method_remote.remote(matrix, convergence) for matrix in matrices]
            else:
                processing_tasks = [process_matrix_file_all_remote.remote(matrix) for matrix in matrices]
            results = ray.get(processing_tasks)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(
            f"Total execution time for processing {count} files, mode:{mode}, method:{method}, convergence:{convergence}: {execution_time:.2f} seconds")
    ray.shutdown()


if __name__ == "__main__":
    matrix_dir = "/home/ubuntu/mapreduce/matrix"
    # file_counts = [5]
    # file_counts = [25, 50, 100]
    # file_counts = [400]
    file_counts = [25, 50, 100, 200, 400]

    args = sys.argv[1:]
    kwargs = {}
    for arg in args:
        k, v = arg.split('=')
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        kwargs[k] = v
    mode = kwargs.get('mode', 'ray')
    method = kwargs.get('method', 'power_method')
    convergence = kwargs.get('convergence', False)
    run_experiment(matrix_dir, file_counts, mode, method, convergence)