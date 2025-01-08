from Bio import SeqIO

def extract_first_n_sequences(input_fasta, output_fasta, n):
    '''
    Extracts the first n sequences from a FASTA file and saves them to another FASTA file.

    Parameters:
        input_fasta (str): Path to the input FASTA file.
        output_fasta (str): Path to the output FASTA file.
        n (int): Number of sequences to extract.
    '''
    try:
        # Read the sequences from the input FASTA file
        sequences = list(SeqIO.parse(input_fasta, 'fasta'))
        
        # Check if n exceeds the number of available sequences
        if n > len(sequences):
            print(f'Warning: Only {len(sequences)} sequences are available. Extracting all of them.')
            n = len(sequences)

        # Write the first n sequences to the output FASTA file
        SeqIO.write(sequences[:n], output_fasta, 'fasta')
        print(f"Successfully saved the first {n} sequences to '{output_fasta}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_fasta}' was not found.")
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


input_fasta = 'data/dataset_alt_40k/alt_promoter_active_40k_200bp.fa'
output_fasta = 'data/dataset_alt_40k/alt_promoter_active_100_test_seq.fa'
n = 100

extract_first_n_sequences(input_fasta, output_fasta, n)
