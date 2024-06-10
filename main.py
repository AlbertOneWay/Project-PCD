import argparse
from processing import sequential, multithreaded, multiprocessing, mpi
from utils.file import write_to_file
from utils.graficar import graphics
from utils.drawing import plot_metrics	

def calculate_metrics(parallel_times, num_threads_list):
    metrics = []
    baseline_time = parallel_times[0]  # Tiempo de referencia con 1 hilo/proceso
    for i in range(len(parallel_times)):
        if i == 0:
            speedup = 1  # Aceleración base para 1 hilo/proceso
            efficiency = 1  # Eficiencia base para 1 hilo/proceso
        else:
            speedup = baseline_time / parallel_times[i]
            efficiency = speedup / num_threads_list[i]
        metrics.append((num_threads_list[i], parallel_times[i], speedup, efficiency))
    return metrics

def main():

    arg_parser = argparse.ArgumentParser(description='Proceso de análisis de secuencias')

    
    arg_parser.add_argument('--input1', dest='input1', type=str, 
                            default=None, help='Archivo de secuencia en formato FASTA')

    arg_parser.add_argument('--input2', dest='input2', type=str, 
                            default=None, help='Archivo de secuencia en formato FASTA')

    arg_parser.add_argument('--use_sequential', action='store_true', 
                            help='Activar la ejecución secuencial')
    
    arg_parser.add_argument('--use_multithreaded', action='store_true', 
                            help='Activar la ejecución utilizando hilos')

    arg_parser.add_argument('--use_multiprocessing', action='store_true', 
                            help='Activar la ejecución utilizando multiprocessing')

    arg_parser.add_argument('--use_mpi', action='store_true', 
                            help='Activar la ejecución utilizando mpi4py')

    arg_parser.add_argument('--plot_sppedup_mpi', action='store_true', 
                            help='Graficar velocidades y aceleraciones en MPI4PY')

  
    parsed_args = arg_parser.parse_args()

    # Lógica para ejecutar el dotplot en función del modo seleccionado
    if parsed_args.use_sequential:
        print("hola")
        load_time, process_time = sequential.generate_sequential_dotplot(parsed_args.input1, parsed_args.input2)
        write_to_file('results/sequential/results_time_sequential.txt', [f"Tiempo de carga de archivos: {str(load_time)} segundos", f"Tiempo de procesamiento: {str(process_time)} segundos"])
    elif parsed_args.use_multithreaded:
        num_threads_list = [1, 2, 4, 8]
        parallel_times = []

        for num_threads in num_threads_list:
            print(f"Ejecución con {num_threads} hilos")
            load_time, process_time, image_time, total_time = multithreaded.generate_multithreaded_dotplot(parsed_args.input1, parsed_args.input2, num_threads)
            parallel_times.append(process_time)
            write_to_file(f'results/multithreaded/results_time_multithreaded_{num_threads}.txt', [
                f"Tiempo de carga de archivos: {load_time} segundos",
                f"Tiempo de procesamiento: {process_time} segundos",
                f"Tiempo de generación de imagen: {image_time} segundos",
                f"Tiempo total: {total_time} segundos"
            ])

        metrics = calculate_metrics(parallel_times, num_threads_list)

        metrics_content = ["Hilos, Tiempo Paralelo, Aceleración, Eficiencia"]
        for num_threads, parallel_time, speedup, efficiency in metrics:
            metrics_content.append(f"{num_threads}, {parallel_time}, {speedup}, {efficiency}")
        write_to_file('results/multithreaded/metrics_multithreaded.txt', metrics_content)

        for num_threads, parallel_time, speedup, efficiency in metrics:
            print(f"Hilos: {num_threads}, Tiempo Paralelo: {parallel_time} segundos, Aceleración: {speedup}, Eficiencia: {efficiency}")
        
        plot_metrics(metrics, "Multithreaded", "results/multithreaded")
    elif parsed_args.use_multiprocessing:
        num_threads_list = [1, 2, 4, 8]
        parallel_times = []

        for num_threads in num_threads_list:
            print(f"Ejecución con {num_threads} procesos")
            load_time, process_time, image_time, total_time = multiprocessing.generate_multiprocessing_dotplot(parsed_args.input1, parsed_args.input2, num_threads)
            parallel_times.append(process_time)
            write_to_file(f'results/multiprocessing/results_time_multiprocessing_{num_threads}.txt', [
                f"Tiempo de carga de archivos: {load_time} segundos",
                f"Tiempo de procesamiento: {process_time} segundos",
                f"Tiempo de generación de imagen: {image_time} segundos",
                f"Tiempo total: {total_time} segundos"
            ])

        metrics = calculate_metrics(parallel_times, num_threads_list)

        metrics_content = ["Procesos, Tiempo Paralelo, Aceleración, Eficiencia"]
        for num_threads, parallel_time, speedup, efficiency in metrics:
            metrics_content.append(f"{num_threads}, {parallel_time}, {speedup}, {efficiency}")
        write_to_file('results/multiprocessing/metrics_multiprocessing.txt', "\n".join(metrics_content))

        for num_threads, parallel_time, speedup, efficiency in metrics:
            print(f"Procesos: {num_threads}, Tiempo Paralelo: {parallel_time} segundos, Aceleración: {speedup}, Eficiencia: {efficiency}")

        plot_metrics(metrics, "Multiprocessing", "results/multiprocessing")
    elif parsed_args.use_mpi:

        load_time, process_time, image_time, total_time, size, rank = mpi.generate_mpi_dotplot(parsed_args.input1, parsed_args.input2)
        
        if rank == 0:
            write_to_file(f'results/mpi/results_time_mpi_{size}.txt', [
                f"Tiempo de carga de archivos: {load_time} segundos",
                f"Tiempo de procesamiento: {process_time} segundos",
                f"Tiempo de generación de imagen: {image_time} segundos",
                f"Tiempo total: {total_time} segundos"
            ])
            

            print(f"Procesos: {size}, Tiempo de procesamiento: {process_time} segundos")
    
    elif parsed_args.plot_sppedup_mpi:
        graphics()
            
    else:
        print("Por favor, seleccione un modo de ejecución: --sequential_mode, --use_multiprocessing, o --use_mpi")


if __name__ == "__main__":
    main()