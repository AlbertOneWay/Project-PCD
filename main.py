import argparse
from processing import sequential
from utils.file import write_to_file

def main():

    arg_parser = argparse.ArgumentParser(description='Proceso de análisis de secuencias')

    
    arg_parser.add_argument('--input1', dest='input1', type=str, 
                            default=None, help='Archivo de secuencia en formato FASTA')

    arg_parser.add_argument('--input2', dest='input2', type=str, 
                            default=None, help='Archivo de secuencia en formato FASTA')

    arg_parser.add_argument('--use_sequential', action='store_true', 
                            help='Activar la ejecución secuencial')

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
    #elif parsed_args.use_multiprocessing:
        #multiprocessing.generate_dotplot(parsed_args.input1, parsed_args.input2)
    #elif parsed_args.use_mpi:
        #mpi.generate_dotplot(parsed_args.input1, parsed_args.input2, parsed_args.process_count)
    else:
        print("Por favor, seleccione un modo de ejecución: --sequential_mode, --use_multiprocessing, o --use_mpi")


if __name__ == "__main__":
    main()