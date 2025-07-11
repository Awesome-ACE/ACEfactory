#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Use HF mirror if needed
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled


### Dataset configuration
dataset=dataset
dataset_path=data/data_with_sst_token
pdb_type=ESMFold
pooling_head=mean  #attention1d ,light_attention ,mean 

### Training configuration
lr=5e-4
training_method=freeze  # 'full', 'freeze', 'lora', 'plm-lora', 'plm-qlora', 'plm-adalora', 'plm-dora', 'plm-ia3'
batch_token=12000
gradient_accumulation_steps=8
num_epochs=50
patience=5

### Regression task configuration
problem_type=regression
num_labels=1
# All regression evaluation metrics
metrics=spcc,pcc,r2,rmse,mse,mae
# Monitor metric: using mean squared error (lower is better)
monitor=mse
monitor_strategy=min

# Define all models to train
declare -A models=(
    ["facebook/esm2_t6_8M_UR50D"]="ESM2-8M"
    ["facebook/esm2_t12_35M_UR50D"]="ESM2-35M"
    ["facebook/esm2_t30_150M_UR50D"]="ESM2-150M"
    ["facebook/esm2_t33_650M_UR50D"]="ESM2-650M"
    ["facebook/esm1b_t33_650M_UR50S"]="ESM-1b"
    ["facebook/esm1v_t33_650M_UR90S_1"]="ESM-1v-1"
    ["facebook/esm1v_t33_650M_UR90S_2"]="ESM-1v-2"
    ["facebook/esm1v_t33_650M_UR90S_3"]="ESM-1v-3"
    ["facebook/esm1v_t33_650M_UR90S_4"]="ESM-1v-4"
    ["facebook/esm1v_t33_650M_UR90S_5"]="ESM-1v-5"
    ["Rostlab/prot_bert"]="ProtBert-uniref50"
    ["Rostlab/prot_bert_bfd"]="ProtBert-bfd"
    ["Rostlab/prot_t5_xl_bfd"]="ProtT5-xl-bfd"
    ["Rostlab/prot_t5_xl_uniref50"]="ProtT5-xl-uniref50"
    ["ElnaggarLab/ankh-base"]="Ankh-base"
    ["ElnaggarLab/ankh-large"]="Ankh-large"
    ["AI4Protein/ProSST-4096"]="ProSST-4096"
    ["AI4Protein/ProSST-2048"]="ProSST-2048"
    ["AI4Protein/ProSST-1024"]="ProSST-1024"
    ["AI4Protein/ProSST-512"]="ProSST-512"
    ["AI4Protein/ProSST-128"]="ProSST-128"
    ["AI4Protein/ProSST-20"]="ProSST-20"
    ["AI4Protein/Prime_690M"]="ProPrime-690M"
    ["AI4Protein/ProPrime_650M_OGT_Prediction"]="ProPrime-650M-OGT"
    ["AI4Protein/deep_base"]="Deep_base"
    ["AI4Protein/deep_bpe_50"]="Deep_BPE_50"
    ["AI4Protein/deep_bpe_100"]="Deep_BPE_100"
    ["AI4Protein/deep_bpe_200"]="Deep_BPE_200"
    ["AI4Protein/deep_bpe_400"]="Deep_BPE_400"
    ["AI4Protein/deep_bpe_800"]="Deep_BPE_800"
    ["AI4Protein/deep_bpe_1600"]="Deep_BPE_1600"
    ["AI4Protein/deep_bpe_3200"]="Deep_BPE_3200"
    ["AI4Protein/deep_unigram_50"]="Deep_Unigram_50"
    ["AI4Protein/deep_unigram_100"]="Deep_Unigram_100"
    ["AI4Protein/deep_unigram_200"]="Deep_Unigram_200"
    ["AI4Protein/deep_unigram_400"]="Deep_Unigram_400"
    ["AI4Protein/deep_unigram_800"]="Deep_Unigram_800"
    ["AI4Protein/deep_unigram_1600"]="Deep_Unigram_1600"
    ["AI4Protein/deep_unigram_3200"]="Deep_Unigram_3200"
)

# Record start time
start_time=$(date +%s)

echo "=========================================="
echo "Starting batch training for ${#models[@]} models"
echo "=========================================="

# Create log directory
log_dir="logs/batch_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $log_dir

# Train all models
model_count=0
for model_path in "${!models[@]}"; do
    model_count=$((model_count + 1))
    model_name=${models[$model_path]}
    
    # Extract plm_source and plm_model from path
    plm_source=$(echo $model_path | cut -d'/' -f1)
    plm_model=$(echo $model_path | cut -d'/' -f2)
    
    ### Output configuration
    output_dir=debug/$dataset/$plm_model/regression
    output_model_name="$training_method"_"$pdb_type"_regression_lr"$lr"_bt${batch_token}_ga${gradient_accumulation_steps}.pt
    
    echo ""
    echo "=========================================="
    echo "[$model_count/${#models[@]}] Starting training: $model_name"
    echo "Model path: $model_path"
    echo "PLM Source: $plm_source"
    echo "PLM Model: $plm_model"
    echo "Output directory: $output_dir"
    echo "=========================================="
    
    # Record model training start time
    model_start_time=$(date +%s)
    
    # Execute training and save output to log file
    python3 src/train.py \
        --plm_model $plm_source/$plm_model \
        --dataset $dataset \
        --dataset_path $dataset_path\
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
        --max_grad_norm 1.0 \
        

        # 2>&1 | tee "$log_dir/${model_name}.log"
    
    # Check if training was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "[$model_count/${#models[@]}] $model_name training successful!"
        echo "SUCCESS: $model_name" >> "$log_dir/summary.log"
    else
        echo "[$model_count/${#models[@]}] $model_name training failed!"
        echo "FAILED: $model_name" >> "$log_dir/summary.log"
    fi
    
    # Calculate model training time
    model_end_time=$(date +%s)
    model_duration=$((model_end_time - model_start_time))
    echo "Training duration: $((model_duration / 60)) minutes $((model_duration % 60)) seconds"
    echo ""
done

# Calculate total training time
end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo "=========================================="
echo "Batch training completed!"
echo "Total training duration: $((total_duration / 3600)) hours $((total_duration % 3600 / 60)) minutes"
echo "Logs saved in: $log_dir"
echo "=========================================="

# Display training summary
echo ""
echo "Training summary:"
cat "$log_dir/summary.log"


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