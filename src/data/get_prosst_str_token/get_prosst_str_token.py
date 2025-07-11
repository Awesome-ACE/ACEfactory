import os
import argparse
import json
import torch
import pandas as pd
from get_sst_seq import SSTPredictor
import gc
import sys

DEFAULT_VOCAB_SIZES = [20, 128, 512, 1024, 2048, 4096]

def get_prosst_token(pdb_file, processor, structure_vocab_size):
    structure_result = processor.predict_from_pdb(pdb_file)
    pdb_name = os.path.basename(pdb_file)
    sst_seq = structure_result[0][f'{structure_vocab_size}_sst_seq']
    sst_seq = [int(i + 3) for i in sst_seq]
    return sst_seq

def process_csv_with_prosst_tokens(csv_file, pdb_dir, structure_vocab_sizes=DEFAULT_VOCAB_SIZES, output_csv=None):
    df = pd.read_csv(csv_file)
    list_prosst_tokens = {f"stru_token_{vocab_size}": [] for vocab_size in structure_vocab_sizes}

    for idx, row in df.iterrows():
        entry = row['entry']
        pdb_file = os.path.join(pdb_dir, f"{entry}.pdb")
        for vocab_size in structure_vocab_sizes:
            processor = SSTPredictor(structure_vocab_size=vocab_size)
            prosst_tokens = get_prosst_token(pdb_file, processor, vocab_size)
            list_prosst_tokens[f"stru_token_{vocab_size}"].append(prosst_tokens)
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for column_name, tokens in list_prosst_tokens.items():
        df[column_name] = tokens

    if output_csv is None:
        output_csv = csv_file
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProSST structure token generator for CSV')
    parser.add_argument('--pdb_dir', type=str, help='Directory containing PDB files')
    parser.add_argument('--pdb_file', type=str, help='Single PDB file path')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--pdb_index_file', type=str, default=None, help='PDB index file for sharding')
    parser.add_argument('--pdb_index_level', type=int, default=1, help='Directory hierarchy depth')
    parser.add_argument('--error_file', type=str, help='Error log output path')
    parser.add_argument('--out_file', type=str, help='Output JSON file path')
    parser.add_argument('--csv_file', type=str, help='Input CSV file path')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV file path (default: overwrite input)')

    args = parser.parse_args()

    if not args.csv_file:
        print("Error: --csv_file must be provided.")
        sys.exit(1)

    if not args.pdb_dir:
        args.pdb_dir = "/home/lwj/520/pHCENet/dataset/pdb/"

    print("CSV Mode:")
    print(f"  Input CSV: {args.csv_file}")
    print(f"  PDB Directory: {args.pdb_dir}")
    print(f"  Vocabulary sizes: {DEFAULT_VOCAB_SIZES}")

    process_csv_with_prosst_tokens(
        csv_file=args.csv_file,
        pdb_dir=args.pdb_dir,
        structure_vocab_sizes=DEFAULT_VOCAB_SIZES,
        output_csv=args.output_csv
    )
