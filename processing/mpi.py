from mpi4py import MPI
import numpy as np
import time
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter

def worker_mpi(sequence1, sequence2, start_idx, end_idx):
    dotplot = []
    for i in range(start_idx, end_idx):
        row = []
        for j in range(len(sequence2)):
            if sequence1[i] == sequence2[j]:
                if i == j:
                    row.append(1)
                else:
                    row.append(0.7)
            else:
                row.append(0)
        dotplot.append(row)
    return dotplot

def generate_mpi_dotplot(file1, file2, n=4):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Medir tiempo de carga de archivos
        start_load_time = time.time()
        seq1 = read_FASTA(file1)
        seq2 = read_FASTA(file2)

        sequence1 = seq1[0:10000]
        sequence2 = seq2[0:10000]
        end_load_time = time.time()
        load_time = end_load_time - start_load_time
        print(f"Tiempo de carga de archivos: {load_time} segundos")
    else:
        sequence1 = None
        sequence2 = None
        load_time = None

    # Broadcast sequences to all processes
    sequence1 = comm.bcast(sequence1, root=0)
    sequence2 = comm.bcast(sequence2, root=0)

    # Split the task among processes
    chunk_size = len(sequence1) // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank != size - 1 else len(sequence1)

    start_process_time = time.time()

    # Each process computes its part of the dotplot
    local_dotplot = worker_mpi(sequence1, sequence2, start_idx, end_idx)

    # Gather results at root process
    all_dotplots = comm.gather(local_dotplot, root=0)

    if rank == 0:
        # Flatten the list of lists
        dotplot_matrix = [row for sublist in all_dotplots for row in sublist]

        end_process_time = time.time()
        process_time = end_process_time - start_process_time

        start_image_time = time.time()

        dotplot_matrix_np = np.array(dotplot_matrix, dtype=np.uint8)

        # Aplicar el filtro
        apply_custom_filter(dotplot_matrix_np, f"results/mpi/dotplot_filtered.jpg")
        draw_dotplot(dotplot_matrix, f'results/mpi/dotplot.jpg')

        end_image_time = time.time()
        image_time = end_image_time - start_image_time

        total_time = load_time + process_time + image_time
        return load_time, process_time, image_time, total_time
    else:
        return None, None, None, None

