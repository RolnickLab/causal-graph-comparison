#!/bin/bash
#SBATCH --job-name=mlp_inference
#SBATCH --output=logs/mlp_inference_%j.out
#SBATCH --error=logs/mlp_inference_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules (adjust based on your cluster setup)
module load python/3.9
module load cuda/11.7

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set paths (modify these as needed)
CHECKPOINT_PATH="$SCRATCH/mlp_training_results/best_model.pth"
OUTPUT_DIR="$SCRATCH/mlp_inference_results"

# Run the inference script
python scripts/04-test-mlp.py \
    --checkpoint $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 64 \
    --gpu

echo "Inference completed! Results saved to: $OUTPUT_DIR" 