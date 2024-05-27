from Bio import SeqIO

def read_FASTA(file_name):
    sequences = []
    for record in SeqIO.parse(file_name, "fasta"):
        sequences.append(str(record.seq))
    return "".join(sequences)

def write_to_file(filename, content):
    try:
        with open(filename, 'w') as f:
            for text in content:
                f.write(str(text) + "\n")
        print(f"Contenido escrito en el archivo: {filename}")
    except Exception as e:
        print(f"Error escribiendo en el archivo {filename}: {e}")