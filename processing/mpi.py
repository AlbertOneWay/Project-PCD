import time
import numpy as np
from mpi4py import MPI
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter
import argparse

def worker_mpi(i, sequence1, sequence2):
    return [1 if sequence1[i] == sequence2[j] and i == j else 0.7 if sequence1[i] == sequence2[j] else 0 for j in range(len(sequence2))]

def parallel_mpi_dotplot(sequence1, sequence2, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    chunk_size = len(sequence1) // size
    start = rank * chunk_size
    end = start + chunk_size if rank < size - 1 else len(sequence1)

    return [worker_mpi(i, sequence1, sequence2) for i in range(start, end)]

def generate_mpi_dotplot(file1, file2, num_processes):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    start_load_time = time.time()
    seq1 = read_FASTA(file1)
    seq2 = read_FASTA(file2)

    sequence1 = seq1[:10000]
    sequence2 = seq2[:10000]
    end_load_time = time.time()

    load_time = end_load_time - start_load_time
    print(f"Tiempo de carga de archivos: {load_time} segundos")

    start_process_time = time.time()

    dotplot = parallel_mpi_dotplot(sequence1, sequence2, comm)

    if rank == 0:
        dotplot_matrices = comm.gather(dotplot, root=0)
        dotplot_matrix = np.concatenate(dotplot_matrices)

        end_process_time = time.time()
        process_time = end_process_time - start_process_time

        start_image_time = time.time()

        dotplot_matrix_np = np.array(dotplot_matrix, dtype=np.uint8)

        apply_custom_filter(dotplot_matrix_np, "results/mpi/dotplot_filtered.jpg")
        draw_dotplot(dotplot_matrix, "results/mpi/dotplot.jpg")

        end_image_time = time.time()
        image_time = end_image_time - start_image_time

        total_time = load_time + process_time + image_time
        return load_time, process_time, image_time, total_time
    else:
        comm.gather(dotplot, root=0)
        return None


