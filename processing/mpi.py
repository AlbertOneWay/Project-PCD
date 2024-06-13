from mpi4py import MPI
import numpy as np
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter
import time

def worker(sequence1_chunk, sequence2_numeric):
    dotplot_chunk = np.equal.outer(sequence1_chunk, sequence2_numeric).astype(np.uint8)
    return dotplot_chunk

def generate_mpi_dotplot(file1, file2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_load_time = time.time()
        seq1 = read_FASTA(file1)
        seq2 = read_FASTA(file2)

        sequence1 = seq1[0:20000]
        sequence2 = seq2[0:20000]
        end_load_time = time.time()

        load_time = end_load_time - start_load_time
        print(f"Tiempo de carga de archivos: {load_time} segundos")

        base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        sequence1_numeric = np.array([base_mapping[base] for base in sequence1], dtype=np.uint8)
        sequence2_numeric = np.array([base_mapping[base] for base in sequence2], dtype=np.uint8)

        len_seq1 = len(sequence1_numeric)
        len_seq2 = len(sequence2_numeric)

        chunk_size = len_seq1 // size
    else:
        sequence1_numeric = None
        sequence2_numeric = None
        chunk_size = None
        len_seq1 = None

    sequence1_numeric = comm.bcast(sequence1_numeric, root=0)
    sequence2_numeric = comm.bcast(sequence2_numeric, root=0)
    chunk_size = comm.bcast(chunk_size, root=0)
    len_seq1 = comm.bcast(len_seq1, root=0)

    start = rank * chunk_size
    end = len_seq1 if rank == size - 1 else (rank + 1) * chunk_size
    sequence1_chunk = sequence1_numeric[start:end]

    start_process_time = time.time()
    dotplot_chunk = worker(sequence1_chunk, sequence2_numeric)
    end_process_time = time.time()

    process_time = end_process_time - start_process_time

    dotplot_matrix = None
    if rank == 0:
        dotplot_matrix = np.zeros((len_seq1, len_seq2), dtype=np.uint8)

    comm.Gatherv(sendbuf=dotplot_chunk, recvbuf=dotplot_matrix, root=0)

    if rank == 0:
        apply_custom_filter(dotplot_matrix, f"results/mpi/dotplot_filtered{size}.jpg")
        draw_dotplot(dotplot_matrix, f'results/mpi/dotplot{size}.jpg')

        total_time = load_time + process_time
        return load_time, process_time, total_time, size, rank
    else:
        return None, None, None, None, None