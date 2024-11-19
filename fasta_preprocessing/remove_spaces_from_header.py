import argparse

def transform_header(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.startswith(">"):
                parts = line.strip().split()
                # Extract the desired components
                chrom = parts[1].replace("chr", "")
                position = parts[2]
                transcript = parts[5]
                gb = parts[6]
                snps = parts[7]
                # Construct the new header
                new_header = f">{chrom}:{position}_{transcript}_{gb}_{snps}"
                outfile.write(new_header + "\n")
            else:
                # Write the sequence lines as is
                outfile.write(line)

def main():
    parser = argparse.ArgumentParser(description="Transform FASTA headers into a new format.")
    parser.add_argument("input_file", help="Path to the input FASTA file.")
    parser.add_argument("output_file", help="Path to the output FASTA file.")
    args = parser.parse_args()
    
    transform_header(args.input_file, args.output_file)

if __name__ == "__main__":
    main()