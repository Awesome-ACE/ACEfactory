import os
import re
import pandas as pd
from pathlib import Path
import argparse
import sys


def extract_metrics_from_log(log_file_path):
    """
    Extract regression metrics from log files
    Returns a dictionary containing 6 metrics
    """
    metrics = {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Define metric patterns - corrected to English format and support negative numbers
        patterns = {
            'spcc': r'Test spcc:\s*([-]?[\d.]+)',
            'pcc': r'Test pcc:\s*([-]?[\d.]+)',
            'r2': r'Test r2:\s*([-]?[\d.]+)',
            'rmse': r'Test rmse:\s*([-]?[\d.]+)',
            'mse': r'Test mse:\s*([-]?[\d.]+)',
            'mae': r'Test mae:\s*([-]?[\d.]+)'
        }
        
        # Extract each metric
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                # Take the last matching value (latest result)
                metrics[metric_name] = float(matches[-1])
            else:
                metrics[metric_name] = None
                
    except Exception as e:
        print(f"Error reading file {log_file_path}: {e}")
        # If error occurs, return null values
        metrics = {key: None for key in ['spcc', 'pcc', 'r2', 'rmse', 'mse', 'mae']}
    
    return metrics

def main():
    """
    Main function: traverse all subdirectories, extract metrics and generate CSV
    """
    parser = argparse.ArgumentParser(description='Extract regression metrics from logs')
    parser.add_argument('--log_dir', type=str, help='Log directory path to save summary')
    parser.add_argument('--work_dir', type=str, default='ckpt/debug/phmax2', help='Working directory to scan')
    parser.add_argument('--problem_type', type=str, required=True, help='Problem type (e.g., regression)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., phmax2)')
    parser.add_argument('--training_method', type=str, default="", help='Training method (e.g., freeze)')

    args = parser.parse_args()
    
    current_dir = Path(args.work_dir)
    
    print("Starting to scan subdirectories...")
    print(f"Working directory: {current_dir}")
    print(f"Absolute path: {current_dir.absolute()}")
    results = []
    
    print("Starting to scan subdirectories...")
    
    # Traverse all subdirectories in current directory
    for subdir in current_dir.iterdir():
        if subdir.is_dir():
            # Check if regression directory exists
            regression_dir = subdir / 'regression'
            if regression_dir.exists() and regression_dir.is_dir():
                # Look for .log files
                log_files = list(regression_dir.glob('*.log'))
                
                if log_files:
                    # If there are multiple .log files, take the first one
                    log_file = log_files[0]
                    print(f"Processing {subdir.name} - {log_file.name}")
                    
                    # Extract metrics
                    metrics = extract_metrics_from_log(log_file)
                    
                    # Add directory name
                    metrics['directory'] = subdir.name
                    results.append(metrics)
                    
                    # Print extracted metrics
                    print(f"  spcc: {metrics['spcc']}, pcc: {metrics['pcc']}, r2: {metrics['r2']}")
                    print(f"  rmse: {metrics['rmse']}, mse: {metrics['mse']}, mae: {metrics['mae']}")
                else:
                    print(f"No .log file found in {subdir.name}/regression/")
            else:
                print(f"No regression directory in subdirectory {subdir.name}")
    
    if results:
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Set directory name as row index
        df.set_index('directory', inplace=True)
        
        # Reorder columns
        column_order = ['spcc', 'pcc', 'r2', 'rmse', 'mse', 'mae']
        df = df[column_order]
        
        # Save as CSV
        #output_file = 'regression_metrics.csv'
        output_file =Path(args.log_dir) /f'{args.problem_type}_{args.dataset}_{args.training_method}_metrics.csv'
        df.to_csv(output_file, encoding='utf-8')
        
        print(f"\nSuccessfully generated CSV file: {output_file}")
        print(f"Processed {len(results)} subdirectories in total")
        
        print("\nPreview results:")
        
        print(df)

        summary_log_path = Path(args.log_dir) / f'{args.problem_type}_{args.dataset}_{args.training_method}_metrics.log'
        with open(summary_log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write("Full Metrics Table (copied from CSV):\n")
            f.write("="*60 + "\n")
            f.write(df.to_string())
            f.write("\n" + "="*60 + "\n")
        print(f"Saved to {summary_log_path}")
        
    else:
        print("No valid log files found")

if __name__ == "__main__":
    main()