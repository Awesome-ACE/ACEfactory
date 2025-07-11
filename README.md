# ACEfactory
A comprehensive pipeline for protein property prediction using multiple Protein Language Models (PLMs) and structure-aware features.
# Protein Property Prediction Pipeline

A comprehensive pipeline for protein property prediction using multiple Protein Language Models (PLMs) and structure-aware features.

## Overview

This project provides an end-to-end automated pipeline for protein property prediction, integrating state-of-the-art protein structure prediction methods with 38 different pre-trained protein language models. Starting from simple sequence data, the pipeline automatically obtains protein structures, extracts structural features, trains multiple models, and performs systematic performance evaluation.
<img width="1182" height="539" alt="image" src="https://github.com/user-attachments/assets/f2db8588-8369-47c3-aba6-88c61a94ec9a" />


## Features

- **Dual Structure Acquisition Strategy**: Combines AlphaFold database downloads with ESMFold local predictions
- **Multi-Model Evaluation**: Systematic comparison of 38 protein language models
- **Automated Pipeline**: From data preparation to model evaluation
- **Parallel Processing**: Efficient computation with GPU acceleration
- **Standardized Evaluation**: Comprehensive metrics for regression tasks

## Pipeline Workflow

### 1. Data Preparation

Input CSV format with three columns:
- `entry`: Protein identifier
- `aa_seq`: Amino acid sequence  
- `label`: Target property value (for regression)

### 2. Structure Acquisition

#### 2.1 AlphaFold Database Download (`1_1_get_pdb_by_AlphafoldDB.py`)
```bash
python 1_1_get_pdb_by_AlphafoldDB.py
```
- Downloads high-quality structures from AlphaFold database
- Adds `AFDB` column to track download status
- Provides download statistics and error logging

#### 2.2 ESMFold Prediction (`1_2_get_pdb_by_ESMFold.py`)
```bash
python 1_2_get_pdb_by_ESMFold.py
```
- Predicts structures for entries missing in AlphaFold database
- GPU-accelerated structure prediction
- Validates sequence integrity
- Adds `ESMFold` column for prediction status

### 3. Structure Feature Extraction (`2_get_sst_seq.sh`)

```bash
bash 2_get_sst_seq.sh
```
- Extracts ProSST (Protein Structure Sequence Tokenization) features
- Parallel processing for train/validation/test sets
- Disconnection-resistant execution with nohup

### 4. Model Training and Evaluation

#### 4.1 Batch Training (`3_train_compare.sh`)
```bash
bash 3_train_compare.sh
```

Evaluates 38 protein language models including:
- **ESM Series**: ESM2 (8M-650M), ESM-1b, ESM-1v variants
- **ProtBert Series**: UniRef50 and BFD variants
- **ProtT5 Series**: XL-scale models
- **Ankh Series**: Base and large versions
- **ProSST Series**: Multiple token dimensions (20-4096)
- **ProPrime Series**: Including OGT-specific version
- **Deep Series**: BPE and Unigram tokenization variants

**Training Configuration**:
- Fine-tuning: freeze(default) 
- Target modules: query, key, value
- Learning rate: 5e-4
- Gradient accumulation: 8 steps
- Evaluation metrics: SPCC, PCC, R², RMSE, MSE, MAE
- Early stopping: Based on MSE, patience=5

#### 4.2 Single Model Optimization (`4_train_single.sh`)
```bash
bash 4_train_single.sh
```
- Deep optimization of best-performing model
- Parameter tuning and method refinement
- Architecture adjustments for optimal performance

## Installation

### Prerequisites

- CUDA 12.2 (recommended)
- Anaconda or Miniconda
- Git

### Step-by-Step Installation

#### 1. Clone the Repository

First, get the VenusFactory code:

```bash
git clone https://github.com/Awesome-ACE/ACEfactory.git
cd ACEfactory
```

#### 2. Create a Conda Environment

Create a new environment named `acenv` with Python 3.10:

```bash
conda create -n acenv python=3.10
conda activate acenv
```

#### 3. Install PyTorch and PyG Dependencies

Install PyTorch with CUDA 12.1 support:

```bash
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyG dependencies
pip install torch_geometric==2.6.1 -f https://pytorch-geometric.com/whl/torch-2.5.1+cu121.html
pip install --no-index torch_scatter==2.1.2 -f https://pytorch-geometric.com/whl/torch-2.5.1+cu121.html
```

#### 4. Install Remaining Dependencies

Install the remaining dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Verify Installation

To verify that everything is installed correctly:

```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"
```

### Troubleshooting

If you encounter any issues:

- **CUDA Version Mismatch**: Make sure your system CUDA version is compatible with PyTorch's CUDA version
- **Missing Dependencies**: Check the error message and install missing packages individually
- **Permission Issues**: Try using `pip install --user` if you don't have system-wide permissions

## Usage

### Quick Start

1. Prepare your data in CSV format with columns: `entry`, `aa_seq`, `label`

2. Run the complete pipeline:
```bash
# Step 1: Get protein structures
python 1_1_get_pdb_by_AlphafoldDB.py --csv_file your_data.csv
python 1_2_get_pdb_by_ESMFold.py  # For missing structures

# Step 2: Extract structural features
bash 2_get_sst_seq.sh

# Step 3: Train and compare models
bash 3_train_compare.sh

# Step 4: Optimize best model
bash 4_train_single.sh
```

### Configuration

Modify the shell scripts to adjust:
- Dataset paths
- Model selection
- Training hyperparameters
- GPU allocation
- Output directories

## Project Structure

```
.
├── data/
│   ├── source_data/          # Original CSV files
│   └── data_with_sst_token/  # Enhanced data with structural features
├── pdbs/                     # Downloaded/predicted PDB files
├── src/
│   ├── train.py             # Main training script
│   └── utils/
│       └── compare.py       # Model comparison utilities
├── logs/                    # Training logs and summaries
├── ckpt/                    # Model checkpoints
├──1_1_get_pdb_by_AlphafoldDB.py
├──1_2_get_pdb_by_ESMFold.py
├──2_get_sst_seq.sh
├──3_train_compare.sh
├──4_train_single.sh
└──requirements.txt
```

## Results

The pipeline automatically generates:
- Performance metrics for all 39 models
- Comparative analysis in CSV format
- Training logs and summaries
- Best model checkpoints for deployment

## Applications

The optimized models can be applied to:
- Guide wet-lab experiment design
- Accelerate computational biology tasks
- Support protein engineering efforts
- Predict various protein properties for drug discovery

## Acknowledgments

This project builds upon the excellent work by **venusfactory**. We gratefully acknowledge their contributions to the field.

Special thanks to the following contributors:
- **Boya Zhang** (Ariel)
- **Wenjie Li** (Cynthia)  
- **Shantong Hu** (Ezreal) 
Their invaluable support and contributions made this project possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support, please open an issue on GitHub or contact [1666521379@qq.com].
