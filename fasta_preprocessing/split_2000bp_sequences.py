import argparse
from Bio import SeqIO

def split_fasta(input_file, output_file):
    with open(output_file, "w") as out_fasta:
        for record in SeqIO.parse(input_file, "fasta"):
            seq = str(record.seq)
            if len(seq) != 2000:
                print(f"Skipping sequence {record.id} (length not 2000 bp).")
                continue
            
            for i in range(10):
                start = i * 200
                end = start + 200
                fragment_seq = seq[start:end]
                fragment_id = f"{record.id}_part{i+1}"
                fragment_record = f">{fragment_id}\n{fragment_seq}\n"
                out_fasta.write(fragment_record)

def main():
    parser = argparse.ArgumentParser(description="Split 2000 bp sequences in a FASTA file into 10 parts of 200 bp.")
    parser.add_argument("input_file", help="Path to the input FASTA file.")
    parser.add_argument("output_file", help="Path to the output FASTA file.")
    args = parser.parse_args()
    
    split_fasta(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
