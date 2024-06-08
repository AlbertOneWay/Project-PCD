from mpi4py import MPI
import numpy as np
import time
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter

def worker_mpi(sequence1_chunk, sequence2):
    dotplot = []
    for i, seq1_char in enumerate(sequence1_chunk):
        row = []
        for j, seq2_char in enumerate(sequence2):
            if seq1_char == seq2_char:
                if i == j:
                    row.append(1)
                else:
                    row.append(0.7)
            else:
                row.append(0)
        dotplot.append(row)
    return dotplot

def generate_mpi_dotplot(file1, file2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_load_time = time.time()
        seq1 = read_FASTA(file1)
        seq2 = read_FASTA(file2)

        sequence1 = seq1[:10000]
        sequence2 = seq2[:10000]
        end_load_time = time.time()
        load_time = end_load_time - start_load_time
        print(f"Tiempo de carga de archivos: {load_time} segundos")

        chunk_size = len(sequence1) // size
        extra = len(sequence1) % size
        chunk_sizes = [chunk_size + 1 if i < extra else chunk_size for i in range(size)]
        chunks = [sequence1[sum(chunk_sizes[:i]):sum(chunk_sizes[:i + 1])] for i in range(size)]
    else:
        sequence2 = None
        chunk_sizes = None
        chunks = None
        load_time = None

    sequence2 = comm.bcast(sequence2, root=0)
    chunk_sizes = comm.bcast(chunk_sizes, root=0)

    local_chunk = comm.scatter(chunks, root=0)

    start_process_time = time.time()

    local_dotplot = worker_mpi(local_chunk, sequence2)

    all_dotplots = comm.gather(local_dotplot, root=0)

    if rank == 0:
        dotplot_matrix = [row for sublist in all_dotplots for row in sublist]

        end_process_time = time.time()
        process_time = end_process_time - start_process_time

        start_image_time = time.time()

        dotplot_matrix_np = np.array(dotplot_matrix, dtype=np.uint8)

        # Aplicar el filtro
        apply_custom_filter(dotplot_matrix_np, f"results/mpi/dotplot_filtered{size}.jpg")
        draw_dotplot(dotplot_matrix, f'results/mpi/dotplot{size}.jpg')

        end_image_time = time.time()
        image_time = end_image_time - start_image_time

        total_time = load_time + process_time + image_time
        return load_time, process_time, image_time, total_time, size, rank
    else:
        return None, None, None, None, None, None