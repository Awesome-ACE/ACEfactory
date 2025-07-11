import pandas as pd
import requests
import os
from pathlib import Path
import time
from tqdm import tqdm

def download_alphafold_structures(csv_file, output_dir='pdbs'):
    """
    Download protein structure files from AlphaFold database
    
    Parameters:
    csv_file: Path to CSV file
    output_dir: Output directory name, default is 'pdbs'
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Check if uniprot_entry column exists
    if 'uniprot_entry' not in df.columns:
        raise ValueError("'uniprot_entry' column not found in CSV file")
    
    # Add AFDB column, initialize all values to False
    df['AFDB'] = False
    
    # Statistics
    total_entries = len(df)
    downloaded = 0
    failed = 0
    skipped = 0
    
    # Download progress bar
    with tqdm(total=total_entries, desc="Download Progress") as pbar:
        for idx, row in df.iterrows():
            uniprot_id = row['uniprot_entry']
            
            # Check if uniprot_entry is a valid value
            if pd.isna(uniprot_id) or uniprot_id == False or uniprot_id == 'false' or uniprot_id == '':
                skipped += 1
                pbar.update(1)
                continue
            
            # Construct download URL
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            output_file = os.path.join(output_dir, f"{uniprot_id}.pdb")
            
            try:
                # Send download request
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Save file
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Update status
                    df.at[idx, 'AFDB'] = True
                    downloaded += 1
                    #tqdm.write(f"✓ Successfully downloaded: {uniprot_id}")
                else:
                    failed += 1
                    #tqdm.write(f"✗ Download failed: {uniprot_id} (Status code: {response.status_code})")
                    
            except requests.exceptions.RequestException as e:
                failed += 1
                tqdm.write(f"✗ Download error: {uniprot_id} - {str(e)}")
            
            # Update progress bar
            pbar.update(1)
            
            # Add small delay to avoid too frequent requests
            time.sleep(0.1)
    
    # Save updated CSV file
    output_csv = csv_file.replace('.csv', '_with_AFDB.csv')
    df.to_csv(output_csv, index=False)
    
    # Print statistics
    print("\n" + "="*50)
    print("Download completed! Statistics:")
    print(f"Total entries: {total_entries}")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Download failed: {failed}")
    print(f"Skipped entries: {skipped}")
    print(f"Updated CSV file saved as: {output_csv}")
    print("="*50)
    
    return df

# Usage example
if __name__ == "__main__":
    # Please replace 'your_file.csv' with your CSV filename
    csv_filename = 'xxx.csv'
    
    # Execute download
    updated_df = download_alphafold_structures(csv_filename)
    
    # Optional: View failed download entries
    failed_entries = updated_df[updated_df['AFDB'] == False]
    if len(failed_entries) > 0:
        print("\nEntries that failed to download:")
        print(failed_entries[['uniprot_entry', 'AFDB']])