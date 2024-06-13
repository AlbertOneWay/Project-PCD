
# Análisis de Rendimiento de Dotplot Secuencial vs. Paralelización

Este documento presenta un análisis de rendimiento de diversas implementaciones del dotplot, una herramienta gráfica crucial en bioinformática para la comparación de secuencias de ADN y proteínas. El estudio abarca cinco enfoques diferentes: secuencial, multihilos, multiprocessing, MPI utilizando la biblioteca mpi4py y PyCUDA. Se busca implementar y comparar el rendimiento de estos métodos mediante diversas métricas, incluyendo tiempo de ejecución total, tiempo de carga de datos, eficiencia y escalabilidad. El objetivo es proporcionar una visión clara y comparativa de cómo diferentes técnicas de paralelización impactan en el rendimiento del dotplot, ofreciendo una guía para optimizar el uso de recursos computacionales en futuras aplicaciones bioinformáticas. Se utilizaron secuencias del E-Coli y la Salmonella como datos de prueba para asegurar un análisis robusto y significativo.


## Inicio

## Prerequisitos
NOTA: Para que se pueda instalar la libreria pycuda es necesario tener ya instalado cuda toolkit.

### 1. Creamos el entrono virtual

virtualenv venv

### 2. Activamos el entorno virtual

-----------------------------

En bash:<br>
source venv/Scripts/activate

En cmd:<br>
.\venv\Scripts\activate

-----------------------------

### 3. Instalamos las librerias necesarias
```bash
pip install -r "requirements.txt"
```
## Como ejecutar

### A. Ejecución Secuencial
Para ejecutar el análisis de secuencias de manera secuencial, utiliza el siguiente comando:

```bash
python main.py --input1 <ruta_a_tu_archivo_fasta1> --input2 <ruta_a_tu_archivo_fasta2> --use_sequential
```

### B. Ejecución multithreading
Para ejecutar el análisis de secuencias utilizando multithreading, utiliza el siguiente comando:

```bash
python main.py --input1 <ruta_a_tu_archivo_fasta1> --input2 <ruta_a_tu_archivo_fasta2> --use_multithreaded
```
Este método ejecutará el análisis utilizando diferentes números de hilos (1, 2, 4, 8) y generará los resultados correspondientes.

### C. Ejecución con Multiprocessing
Para ejecutar el análisis de secuencias utilizando multiprocessing, utiliza el siguiente comando:

```bash
python main.py --input1 <ruta_a_tu_archivo_fasta1> --input2 <ruta_a_tu_archivo_fasta2> --use_multiprocessing
```
Este método ejecutará el análisis utilizando diferentes números de procesos (1, 2, 4, 8) y generará los resultados correspondientes.

### D. Ejecución con MPI
Para ejecutar el análisis de secuencias utilizando MPI, asegúrate de tener mpiexec instalado y utiliza el siguiente comando:

```bash
mpiexec -n <num_procesos> python main.py --input1 <ruta_a_tu_archivo_fasta1> --input2 <ruta_a_tu_archivo_fasta2> --use_mpi
```

### E. Ejecución con PyCUDA
Para ejecutar el análisis de secuencias utilizando PyCUDA, utiliza el siguiente comando:

```bash
python main.py --input1 <ruta_a_tu_archivo_fasta1> --input2 <ruta_a_tu_archivo_fasta2> --use_pycuda
```

### F. Opciones Adicionales
Para graficar las velocidades y aceleraciones en MPI, utiliza:
```bash
python main.py --plot_speedup_mpi
```
Para graficar los tiempos entre los procesos, utiliza:
```bash
python main.py --plot_times_process
```
Ejemplo de Uso
```bash
python main.py --input1 data/sequence1.fasta --input2 data/sequence2.fasta --use_multithreaded
```
Este comando ejecutará el análisis utilizando multithreading y generará los archivos de resultados en el directorio results/multithreaded.

## Estructura de Archivos
* data/: Directorio donde se almacenan los archivos de entrada en formato FASTA.
* processing/: Directorio donde están las diferentes versiones para generar el dotplot.
** results/: Directorio donde se almacenarán los resultados de las ejecuciones.
** mpi/
** multiprocessing/
** multithreaded/
** pycuda/
** sequential/
** utils/: Directorio con los archivos que tienen funciones utilizadas en la mayoria de versiones (Cargar archivos, crear imagen, filtrar imagen, graficas)

* Asegúrate de ajustar las rutas de los archivos según tu estructura de directorios.
