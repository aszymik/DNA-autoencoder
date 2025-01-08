import argparse
from Bio import SeqIO

def convert_sequences_to_uppercase(input_fasta, output_fasta):
    """
    Converts all sequences in a FASTA file to uppercase and writes them to an output file.

    Parameters:
        input_fasta (str): Path to the input FASTA file.
        output_fasta (str): Path to the output FASTA file.
    """
    try:
        # Read sequences, convert to uppercase, and save to output file
        with open(output_fasta, "w") as output_handle:
            for record in SeqIO.parse(input_fasta, "fasta"):
                record.seq = record.seq.upper()  # Convert sequence to uppercase
                SeqIO.write(record, output_handle, "fasta")
        print(f"Sequences converted to uppercase and saved to '{output_fasta}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_fasta}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert all sequences in a FASTA file to uppercase.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Input FASTA file path')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output FASTA file path')
    args = parser.parse_args()

    # Convert sequences to uppercase
    convert_sequences_to_uppercase(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
