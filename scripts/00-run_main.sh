#!/bin/bash

#SBATCH --job-name=run_mlp_%j                                        # Set name of job
#SBATCH --output=slurm/train_mlp_output_%j.txt                       # Set location of output file
#SBATCH --error=slurm/train_mlp_error_%j.txt                         # Set location of error file
#SBATCH --gpus-per-task=1                                               # Ask for 2 GPU
#SBATCH --cpus-per-task=12                                              # Ask for 4 CPUs
#SBATCH --ntasks-per-node=1                                             # Ask for 4 CPUs
#SBATCH --nodes=1                                                       # Ask for 4 CPUs
#SBATCH --mem=64G                                                       # Ask for 32 GB of RAM
#SBATCH --time=6:00:00                                                  # The job will run for 2 hours
#SBATCH --partition=long                                                # Ask for long partition

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10


# 2. Load your environment assuming environment is called "env_climatem" in $HOME/env/ (standardized)
source $HOME/env/env_climatem/bin/activate
# 3. Enable expandable allocator to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== running main.py"

python $HOME/dev/causal-graph-comparison/scripts/00-main.py "$@"