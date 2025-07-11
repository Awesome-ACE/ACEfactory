import pandas as pd
import torch
import esm
import os
import glob
from tqdm import tqdm

# Configuration parameters
INPUT_FOLDER = "input_csvs"  # Input CSV folder path
OUTPUT_FOLDER = "output_results"  # Output results folder path
PDB_BASE_FOLDER = "pdbs"  # PDB files base folder

# Initialize ESMFold model
print("Loading ESMFold model...")
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()  # Change to .cpu() if no GPU available

# Define allowed amino acids
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

# Create output folders
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PDB_BASE_FOLDER, exist_ok=True)

# Get all CSV files
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

if not csv_files:
    print(f"Error: No CSV files found in {INPUT_FOLDER}")
    exit(1)

print(f"Found {len(csv_files)} CSV files to process:")
for csv_file in csv_files:
    print(f"  - {os.path.basename(csv_file)}")

# Global statistics variables
total_files_processed = 0
total_entries_all_files = 0
total_predictions_all_files = 0
total_success_all_files = 0
total_errors_all_files = 0

print("\nStarting batch structure prediction...")

# Process each CSV file
for csv_file in csv_files:
    print(f"\n{'='*60}")
    filename = os.path.basename(csv_file)
    file_prefix = os.path.splitext(filename)[0]
    print(f"Processing: {filename}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if 'aa_seq' not in df.columns:
            print(f"Warning: 'aa_seq' column not found in {filename}, skipping...")
            continue
            
        if 'entry' not in df.columns:
            print(f"Warning: 'entry' column not found in {filename}, skipping...")
            continue
            
        # Add ESMFold column, initialize as False
        df['ESMFold'] = False
        
        # Check if AFDB column exists, if not create it with default False values
        if 'AFDB' not in df.columns:
            df['AFDB'] = False
            print(f"Note: 'AFDB' column not found in {filename}, created with default False values")
        
        # Create separate PDB folder for each file
        pdb_folder = os.path.join(PDB_BASE_FOLDER, file_prefix)
        os.makedirs(pdb_folder, exist_ok=True)
        
        # File-level statistics variables
        list_error = []
        success_count = 0
        total_predictions = 0
        
        print(f"Total entries in file: {len(df)}")
        
        # Iterate and predict structures
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {filename}"):
            # Only predict for entries where AFDB is False (removed sequence length restriction)
            if row['AFDB'] == False:
                total_predictions += 1
                entry = row['entry']
                aa_seq = str(row['aa_seq'])
                
                # Check if sequence is empty or NaN
                if pd.isna(aa_seq) or aa_seq.strip() == '' or aa_seq == 'nan':
                    error_msg = f"Entry {entry}: Empty or invalid sequence"
                    list_error.append(error_msg)
                    # Keep ESMFold as False for failed prediction
                    continue
                
                # Validate sequence contains only valid amino acids
                if not set(aa_seq).issubset(valid_amino_acids):
                    invalid_chars = set(aa_seq) - valid_amino_acids
                    error_msg = f"Entry {entry}: Invalid amino acids found: {invalid_chars}"
                    list_error.append(error_msg)
                    # Keep ESMFold as False for failed prediction
                    continue
                
                try:
                    # Use ESMFold to predict structure
                    with torch.no_grad():
                        output = model.infer_pdb(aa_seq)
                    
                    # Save PDB file
                    pdb_filename = os.path.join(pdb_folder, f"{entry}.pdb")
                    with open(pdb_filename, "w") as f:
                        f.write(output)
                    
                    # Update ESMFold column to True for successful prediction
                    df.at[index, 'ESMFold'] = True
                    success_count += 1
                    
                except Exception as e:
                    # Record error
                    error_msg = f"Entry {entry}: {str(e)}"
                    list_error.append(error_msg)
                    # ESMFold column remains False for failed prediction
        
        # Save updated CSV file
        output_filename = os.path.join(OUTPUT_FOLDER, f"{file_prefix}_ESMFold.csv")
        df.to_csv(output_filename, index=False)
        
        # Save error log
        if list_error:
            error_log_filename = os.path.join(OUTPUT_FOLDER, f"{file_prefix}_esmfold_errors.log")
            with open(error_log_filename, "w") as f:
                for error in list_error:
                    f.write(error + "\n")
        
        # Print file-level statistics
        print(f"\n--- {filename} Summary ---")
        print(f"Total entries: {len(df)}")
        print(f"Predictions attempted: {total_predictions}")
        print(f"Successful predictions: {success_count}")
        print(f"Failed predictions: {len(list_error)}")
        print(f"Success rate: {success_count/total_predictions*100:.1f}%" if total_predictions > 0 else "No predictions attempted")
        print(f"Updated CSV saved as: {output_filename}")
        if list_error:
            print(f"Error log saved as: {error_log_filename}")
        
        # Display ESMFold column statistics
        print(f"ESMFold column statistics:")
        print(df['ESMFold'].value_counts())
        
        # Update global statistics
        total_files_processed += 1
        total_entries_all_files += len(df)
        total_predictions_all_files += total_predictions
        total_success_all_files += success_count
        total_errors_all_files += len(list_error)
        
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        continue

# Print global statistics
print(f"\n{'='*60}")
print(f"=== BATCH PROCESSING SUMMARY ===")
print(f"Files processed: {total_files_processed}/{len(csv_files)}")
print(f"Total entries across all files: {total_entries_all_files}")
print(f"Total predictions attempted: {total_predictions_all_files}")
print(f"Total successful predictions: {total_success_all_files}")
print(f"Total failed predictions: {total_errors_all_files}")
print(f"Overall success rate: {total_success_all_files/total_predictions_all_files*100:.1f}%" if total_predictions_all_files > 0 else "No predictions attempted")
print(f"Results saved in: {OUTPUT_FOLDER}")
print(f"PDB files saved in: {PDB_BASE_FOLDER}")

print(f"\nProcessing completed!")