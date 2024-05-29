import argparse
from processing import sequential, multithreaded, multiprocessing
from utils.file import write_to_file

def calculate_metrics(parallel_times, num_threads_list):
    metrics = []
    for i in range(1, len(parallel_times)):
        speedup = parallel_times[0] / parallel_times[i]
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

    arg_parser.add_argument('--process_count', dest='process_count', type=int, nargs='+', 
                            default=[4], help='Cantidad de procesos para la ejecución con MPI')

  
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

    #elif parsed_args.use_mpi:
        #mpi.generate_dotplot(parsed_args.input1, parsed_args.input2, parsed_args.process_count)
    else:
        print("Por favor, seleccione un modo de ejecución: --sequential_mode, --use_multiprocessing, o --use_mpi")


if __name__ == "__main__":
    main()