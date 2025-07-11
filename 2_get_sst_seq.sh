#!/bin/bash
# Parallel execution script for 3 tasks - disconnection-resistant version
# All tasks run on cuda:0

# Set common parameters
PDB_DIR="pdbs"
PYTHON_SCRIPT="src/data/get_prosst_str_token/get_prosst_str_token.py"

# Define task function
run_task() {
    local csv_file=$1
    local gpu_device=$2
    local task_name=$3
    local output_csv=$4
    
    echo "Starting task: $task_name (GPU: $gpu_device)"
    echo "CSV file: $csv_file"
    echo "Output CSV: $output_csv"
    
    # Set CUDA device and run task
    CUDA_VISIBLE_DEVICES=$gpu_device python $PYTHON_SCRIPT --csv_file "$csv_file" --pdb_dir "$PDB_DIR" --output_csv "$output_csv"
    
    echo "Task completed: $task_name"
}

# Create log directory and output directory
mkdir -p logs
mkdir -p data_with_sst_token

echo "Starting parallel execution of 3 tasks..."
echo "GPU allocation: cuda:0 (3 tasks)"
echo "PDB directory: $PDB_DIR"
echo "=================================="

# Create output directory
mkdir -p data_with_sst_token

# Run 3 tasks on cuda:0 - using nohup to prevent disconnection
nohup bash -c 'run_task() { local csv_file=$1; local gpu_device=$2; local task_name=$3; local output_csv=$4; echo "Starting task: $task_name (GPU: $gpu_device)"; echo "CSV file: $csv_file"; echo "Output CSV: $output_csv"; CUDA_VISIBLE_DEVICES=$gpu_device python data/get_prosst_str_token/get_prosst_str_token.py --csv_file "$csv_file" --pdb_dir "pdbs" --output_csv "$output_csv"; echo "Task completed: $task_name"; }; run_task "data/source_data/train.csv" "0" "train" "data/data_with_sst_token/train.csv"' > logs/train.log 2>&1 &

nohup bash -c 'run_task() { local csv_file=$1; local gpu_device=$2; local task_name=$3; local output_csv=$4; echo "Starting task: $task_name (GPU: $gpu_device)"; echo "CSV file: $csv_file"; echo "Output CSV: $output_csv"; CUDA_VISIBLE_DEVICES=$gpu_device python data/get_prosst_str_token/get_prosst_str_token.py --csv_file "$csv_file" --pdb_dir "pdbs" --output_csv "$output_csv"; echo "Task completed: $task_name"; }; run_task "data/source_data/validation.csv" "0" "validation" "data/data_with_sst_token/validation.csv"' > logs/validation.log 2>&1 &

nohup bash -c 'run_task() { local csv_file=$1; local gpu_device=$2; local task_name=$3; local output_csv=$4; echo "Starting task: $task_name (GPU: $gpu_device)"; echo "CSV file: $csv_file"; echo "Output CSV: $output_csv"; CUDA_VISIBLE_DEVICES=$gpu_device python data/get_prosst_str_token/get_prosst_str_token.py --csv_file "$csv_file" --pdb_dir "pdbs" --output_csv "$output_csv"; echo "Task completed: $task_name"; }; run_task "data/source_data/test.csv" "0" "test" "data/data_with_sst_token/test.csv"' > logs/test.log 2>&1 &

echo "All tasks have been started and are running in the background..."
echo "Log files are saved in the logs/ directory"
echo "Tasks will continue running even if terminal connection is lost"
echo "=================================="

# Get all background process PIDs and save them
echo "Background process PIDs:"
jobs -p > logs/pids.txt
cat logs/pids.txt

echo ""
echo "Commands for monitoring task status:"
echo "  View processes: ps aux | grep get_prosst_str_token.py"
echo "  View logs: tail -f logs/task_name.log"
echo "  View all logs: tail -f logs/*.log"
echo "  Kill all tasks: kill \$(cat logs/pids.txt)"