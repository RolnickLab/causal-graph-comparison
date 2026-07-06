# Thesis: Causal Metrics for Evaluating Deep Learning Climate Emulators
By Christina Isaicu
University of Amsterdam 

🌟Read the thesis [here](Christina_Isaicu_MSc_Thesis.pdf) 🌟

## Installation

```bash
pip install -e . # Install the package in editable mode
```

## Scripts Overview

This repository contains scripts for causal causal graph comparison in climate emulation. The scripts are organized into main execution, evaluation, and plotting.

## Main Execution Scripts

### `scripts/00-main.py`
**Purpose**: Main script that consolidates training, rollouts, and learning causal graphs for different models.

**Arguments**:
- `--difficulty` (str): Difficulty level - choices: ["easy", "med_easy", "med_hard", "hard"]
- `--num_modes` (int): Number of modes - choices: [4, 16, 36, 64]
- `--seed` (int): Random seed - choices: [1, 42, 99], default: 1
- `--dataset_type` (str): Dataset type, default: "savar"
- `--resolution` (int): Resolution per mode, default: 10 (10x10)
- `--model` (str): Model type - choices: ["picabu", "vae", "mlp", "lstm", "cnn"], default: "picabu"

**Usage**:
```bash
python scripts/00-main.py --difficulty hard --num_modes 16 --model mlp --seed 42
```

### `scripts/01-run_main.sh`
**Purpose**: SLURM batch script for running the main script on HPC clusters.

**Features**:
- Requests 1 GPU, 12 CPUs, 64GB RAM
- Uses long partition
- Loads Python 3.10 environment
- Activates climatem environment

**Usage**:
```bash
sbatch scripts/01-run_main.sh --difficulty hard --num_modes 16 --model mlp
```

### `scripts/02-submit_jobs.sh`
**Purpose**: Batch job submission script that runs multiple experiments across different models, difficulties, and modes.

**Configuration**:
- Seeds: 1
- Modes: 4, 16
- Models: mlp, lstm, cnn, vae, picabu
- Difficulties: easy, med_easy, med_hard, hard
- Runtime: 16:00:00

## Evaluation Scripts

### `scripts/03-evaluation-savar.py`
**Purpose**: Evaluates causal discovery and causal representation learning on SAVAR data.

**Key Functions**:
- `eval_savar()`: Main evaluation function
- Compares learned graphs with ground truth
- Computes metrics: SHD, SID, ancestor_aid, oset_aid, parent_aid

**Arguments**:
- `dataset`: Dataset name
- `num_modes`: Number of modes
- `difficulty`: Difficulty level
- `seed`: Random seed

### `scripts/04-evaluation.py`
**Purpose**: General evaluation script for different models trained on SAVAR data.

**Key Functions**:
- `eval()`: Evaluates specific model performance
- Loads model outputs and ground truth data
- Computes various evaluation metrics
- Handles different model types (MLP, CNN, LSTM, VAE)

**Arguments**:
- `model`: Model type
- `dataset`: Dataset name
- `num_modes`: Number of modes
- `difficulty`: Difficulty level
- `seed`: Random seed

## Plotting and Visualization Scripts

### `scripts/05-plotting.py`
**Purpose**: Generates comprehensive plots for evaluation results.

**Features**:
- R² value comparisons across models
- Intervention vs next step RMSE plots
- Model performance visualizations
- Difficulty level comparisons

**Configuration**:
- Loads evaluation results from pickle files
- Supports multiple model types with distinct markers/colors
- Generates publication-ready plots

### `scripts/10-plot-savar.py`
**Purpose**: Specific plotting script for SAVAR data visualization.

**Features**:
- Compares model outputs across different architectures
- Visualizes input vs target vs output relationships
- Generates side-by-side comparisons
- Saves plots to designated output directory

## Dataset and Graph Management Scripts (Part 0, not included in thesis)

We built a class that can generate a dataset of modifications to a causal graph: reverse link direction, missing link, added link, wrong time lag. 
Before applying the metrics to learned graphs, we wanted to systematically understand how AID differs from SHD and F1 when errors occur. 
Unfortunately, due to time constraints analysis of this step was omitted from this work. 

### `scripts/06-part0-dataset-builder.py`
**Purpose**: Builds and visualizes ground truth graphs from SAVAR data.

**Key Functions**:
- `GraphBuilder`: Creates ground truth temporal graphs
- `GraphModifier`: Applies various graph modifications
- Visualizes flattened temporal adjacency graphs
- Supports different graph modification strategies

**Features**:
- Weight randomization
- Time lag modifications
- Graph structure alterations
- Visualization of original vs modified graphs

### `scripts/07-part0-graph-eval.py`
**Purpose**: Evaluates graph modifications and computes graph comparison metrics.

**Key Functions**:
- Applies graph modifications (time lag shifts)
- Computes graph comparison metrics using gadjid
- Evaluates structural differences
- Supports binary graph comparisons

**Metrics Computed**:
- Structural Hamming Distance (SHD)
- Structural Intervention Distance (SID)
- Ancestor, Oset, and Parent AID metrics

## Helper Scripts

### `scripts/08-dataloader-test.py`
**Purpose**: Tests and visualizes the data loading pipeline.

**Features**:
- Tests SAVAR data generation
- Validates data shapes and formats
- Generates test plots
- Verifies data consistency across different configurations

## Configuration Files

- `configs/models.json`: Model configurations
- `configs/savar-picabu.json`: SAVAR-PICABU specific configurations