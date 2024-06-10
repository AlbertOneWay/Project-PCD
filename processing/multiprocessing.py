import time
import numpy as np
import multiprocessing as mp
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter

def worker_multiprocessing(params):
    i, sequence1, sequence2 = params
    row = []
    for j in range(len(sequence2)):
        if sequence1[i] == sequence2[j]:
            if i == j:
                row.append(1)
            else:
                row.append(0.7)
        else:
            row.append(0)
    return row

def parallel_multiprocessing_dotplot(sequence1, sequence2, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as pool:
        dotplot = pool.map(worker_multiprocessing, [(i, sequence1, sequence2) for i in range(len(sequence1))])
    return dotplot

def generate_multiprocessing_dotplot(file1, file2, num_processes=4):
    # Medir tiempo de carga de archivos
    start_load_time = time.time()
    seq1 = read_FASTA(file1)
    seq2 = read_FASTA(file2)

    sequence1 = seq1[0:10000]
    sequence2 = seq2[0:10000]
    end_load_time = time.time()

    load_time = end_load_time - start_load_time
    print(f"Tiempo de carga de archivos: {load_time} segundos")

    start_process_time = time.time()

    # Generar el dotplot en paralelo usando multiprocessing
    dotplot_matrix = parallel_multiprocessing_dotplot(sequence1, sequence2, num_processes)

    end_process_time = time.time()
    process_time = end_process_time - start_process_time

    start_image_time = time.time()

    dotplot_matrix_np = np.array(dotplot_matrix, dtype=np.uint8)

    # Aplicar el filtro
    apply_custom_filter(dotplot_matrix_np, f"results/multiprocessing/dotplot_filtered{num_processes}.jpg")

    draw_dotplot(dotplot_matrix, f'results/multiprocessing/dotplot{num_processes}.jpg')

    end_image_time = time.time()
    image_time = end_image_time - start_image_time

    total_time = load_time + process_time + image_time
    return load_time, process_time, image_time, total_time