#!/bin/bash

#SBATCH --job-name=run_mlp_%j                                        # Set name of job
#SBATCH --gpus-per-task=1                                               # Ask for 1 GPU
#SBATCH --cpus-per-task=12                                              # Ask for 12 CPUs
#SBATCH --ntasks-per-node=1                                             # Ask for 1 node
#SBATCH --nodes=1                                                       # Ask for 1 node
#SBATCH --mem=64G                                                       # Ask for 64 GB of RAM
#SBATCH --time=12:00:00                                                  # The job will run for 12 hours
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