import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def cut_sequence(seq, length):
    assert length <= len(seq)
    n = (len(seq)-length) // 2
    return seq[n:n+length]

def elongate_sequence(seq, length):
    assert length >= len(seq)
    n = (length-len(seq)) // 2
    if len(seq) % 2 == 1:
        seq = 'N' * n + seq + 'N' * (n+1)
    else:                        
        seq = 'N' * n + seq + 'N' * n
    return seq    

def convert_to_specified_length(seq, length=200):
    if len(seq) > length:
        seq = cut_sequence(seq, length)
    else:    
        seq = elongate_sequence(seq, length)
    return seq

def process_fasta_to_specified_length(input_path, output_path, target_length=200, min_length=1):
    sequences = []

    # Read sequences using Biopython
    for record in SeqIO.parse(input_path, "fasta"):
        seq = str(record.seq)
        
        if len(seq) >= min_length:
            # Adjust sequence to specified length
            adjusted_seq = convert_to_specified_length(seq, target_length)
            
            # Create a new SeqRecord with the adjusted sequence
            new_record = SeqRecord(Seq(adjusted_seq), id=record.id, description=record.description)
            sequences.append(new_record)

    # Write adjusted sequences to output file
    with open(output_path, 'w') as output_file:
        SeqIO.write(sequences, output_file, "fasta")


def main():
    parser = argparse.ArgumentParser(description="Process FASTA sequences to a specified length")

    parser.add_argument('-i', '--input_file', type=str, required=True, help='Input FASTA file path')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output FASTA file path')
    parser.add_argument('-l', '--length', type=int, default=200, help='Target length of sequences (default: 200)')
    parser.add_argument('-m', '--min_length', type=int, default=1, help='Minimum length to keep sequences (default: 1)')

    args = parser.parse_args()
    process_fasta_to_specified_length(args.input_file, args.output_file, args.length, args.min_length)


if __name__ == '__main__':
    main()
