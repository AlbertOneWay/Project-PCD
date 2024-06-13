import time
import numpy as np
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter
import gc

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def parallel_cuda_dotplot(sequence1, sequence2):
    N = len(sequence1)
    M = len(sequence2)

    # Tamaño del bloque, max 1024 hilos por bloque (configuración 32x32 = 1024)
    block_size_x = min(32, N)
    block_size_y = min(32, M)

    # Tamaño de la grilla, calculado para cubrir toda la secuencia
    grid_x = (N + block_size_x - 1) // block_size_x
    grid_y = (M + block_size_y - 1) // block_size_y

    # Compilar el código CUDA
    mod = SourceModule("""
        __global__ void dotplot(unsigned char *sequence1, unsigned char *sequence2, unsigned char *dotplot) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
                       
            if (i < %(N)d && j < %(M)d) {
                if (sequence1[i] == sequence2[j]) {
                    dotplot[i * %(M)d + j] = 1;
                } else {
                    dotplot[i * %(M)d + j] = 0;
                }
            }
        }
    """ % {'N': N, 'M': M})

    # Obtener la función del kernel
    kernel = mod.get_function("dotplot")

    # Crear los arreglos de entrada y salida
    sequence1_gpu = drv.mem_alloc(sequence1.nbytes)
    sequence2_gpu = drv.mem_alloc(sequence2.nbytes)
    dotplot_gpu = drv.mem_alloc((N * M * 4))

    # Copiar los datos a la GPU
    drv.memcpy_htod(sequence1_gpu, sequence1)
    drv.memcpy_htod(sequence2_gpu, sequence2)

    # Llamar al kernel
    kernel(sequence1_gpu, sequence2_gpu, dotplot_gpu, block=(block_size_x, block_size_y, 1), grid=(grid_x, grid_y))

    # Copiar los resultados de la GPU
    dotplot = np.zeros((N, M), dtype=np.uint8)
    drv.memcpy_dtoh(dotplot, dotplot_gpu)

    return dotplot

def generate_cuda_dotplot(file1, file2):
    # Medir tiempo de carga de archivos
    start_load_time = time.time()
    seq1 = read_FASTA(file1)
    seq2 = read_FASTA(file2)

    sequence1 = seq1[0:21000]
    sequence2 = seq2[0:21000]

    seq1 = None
    seq2 = None

    file1 = None
    file2 = None
    gc.collect()

    end_load_time = time.time()

    load_time = end_load_time - start_load_time
    print(f"Tiempo de carga de archivos: {load_time} segundos")

    start_process_time = time.time()

    # Convertir secuencias a arrays de números
    base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sequence1_numeric = np.array([base_mapping[base] for base in sequence1], dtype=np.uint8)
    sequence2_numeric = np.array([base_mapping[base] for base in sequence2], dtype=np.uint8)

    sequence1 = None
    sequence2 = None
    gc.collect()

    # Definir número de partes
    num_parts = 1

    # Dividir secuencias en partes
    parts_sequence1 = np.array_split(sequence1_numeric, num_parts)

    sequence1_numeric = None
    gc.collect()

    # Lista para almacenar partes del dotplot
    dotplot_parts = []

    # Generar el dotplot en paralelo usando CUDA para cada parte
    for part_num in range(num_parts):
        dotplot_part = parallel_cuda_dotplot(parts_sequence1[part_num], sequence2_numeric)
        dotplot_parts.append(dotplot_part)
        dotplot_part = None
        gc.collect()

    # Combinar las partes en un solo dotplot
    dotplot_matrix = np.concatenate(dotplot_parts, axis=0, dtype=np.uint8)

    dotplot_parts = None

    gc.collect()


    end_process_time = time.time()
    process_time = end_process_time - start_process_time
    print(f"Tiempo de procesamiento: {process_time} segundos")

    start_image_time = time.time()

    # Dibujar el dotplot
    draw_dotplot(dotplot_matrix, "results/pycuda/dotplot.jpg")

    # Aplicar el filtro
    apply_custom_filter(dotplot_matrix, "results/pycuda/dotplot_filtered.jpg")

    end_image_time = time.time()
    image_time = end_image_time - start_image_time
    print(f"Tiempo de generación de imagen: {image_time} segundos")

    total_time = load_time + process_time + image_time
    print(f"Tiempo total: {total_time} segundos")

    return load_time, process_time, image_time, total_time