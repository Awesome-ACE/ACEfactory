#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Use HF mirror if needed
export HF_ENDPOINT=https://hf-mirror.com

### Dataset configuration
dataset=dataset
dataset_path=data/data_with_sst_token
pdb_type=ESMFold
pooling_head=mean   # attention1d ,light_attention ,mean

### Model configuration
plm_source=AI4Protein
plm_model=ProSST-512

### Training configuration
lr=5e-4
training_method=freeze  # adalora, dora, ia3, lora, qlora
batch_token=12000
gradient_accumulation_steps=8
num_epochs=10
patience=3

### Regression task configuration
problem_type=regression
num_labels=1
# All regression evaluation metrics
metrics=spcc,pcc,r2,rmse,mse,mae
# Monitor metric: using Spearman correlation coefficient (higher is better)
monitor=mse
monitor_strategy=min

### Output configuration
output_dir=debug/$dataset/$plm_model/regression
output_model_name="$training_method"_"$pdb_type"_regression_lr"$lr"_bt${batch_token}_ga${gradient_accumulation_steps}.pt

echo "=========================================="
echo "Starting regression task training"
echo "Dataset: $dataset"
echo "Model: $plm_source/$plm_model"  
echo "Training method: $training_method"
echo "Problem type: $problem_type"
echo "Evaluation metrics: $metrics"
echo "Monitor metric: $monitor ($monitor_strategy)"
echo "Learning rate: $lr"
echo "Output directory: $output_dir"
echo "=========================================="

python3 src/train.py \
    --plm_model $plm_source/$plm_model \
    --dataset $dataset \
    --dataset_path $dataset_path \
    --pdb_type $pdb_type \
    --problem_type $problem_type \
    --num_labels $num_labels \
    --metrics $metrics \
    --monitor $monitor \
    --monitor_strategy $monitor_strategy \
    --learning_rate $lr \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_epochs $num_epochs \
    --batch_token $batch_token \
    --patience $patience \
    --output_dir $output_dir \
    --output_model_name "$output_model_name" \
    --training_method $training_method \
    --lora_target_modules query key value \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --pooling_method $pooling_head \
    --seed 3407 \
    --num_workers 4 \
    --max_grad_norm 1.0

echo "=========================================="
echo "Training completed!"
echo "Model saved at: $output_dir/$output_model_name"
echo "Results recorded in terminal output and log files"
echo "=========================================="

# Run compare.py to extract metrics from debug directory
echo "Starting metrics extraction..."
python src/utils/compare.py \
        --log_dir "$log_dir" \
        --work_dir "ckpt/debug/$dataset"  \
        --problem_type="$problem_type" \
        --dataset="$dataset" \
        --training_method="$training_method"

if [ $? -eq 0 ]; then
    echo "Metrics extraction completed successfully!"
    echo "Results saved to: $log_dir/${problem_type}_${dataset}_${training_method}_metrics.csv"
else
    echo "Metrics extraction failed!"
fi

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "Check the following files for results:"
echo "  - Training logs: $log_dir/"
echo "  - Training summary: $log_dir/summary.log"
echo "  - Training summary: $log_dir/${problem_type}_${dataset}_${training_method}_metrics.log"
echo "  - CSV file: $log_dir/${problem_type}_${dataset}_${training_method}_metrics.csv"
echo "=========================================="