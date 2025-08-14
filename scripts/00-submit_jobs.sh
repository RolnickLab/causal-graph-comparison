# === run chris
seed=1
for mode in 64; do
    for model in vae; do
        for difficulty in easy med_easy med_hard hard; do
            if [[ $model != vae ]]; then
                if [[ $mode -eq 4 ]]; then
                    runtime=12:00:00
                elif [[ $mode -eq 16 ]]; then
                    runtime=12:00:00
                elif [[ $mode -eq 64 ]]; then
                    runtime=30:00:00
                fi
            elif [[ $model = vae ]]; then
                if [[ $mode -eq 4 ]]; then
                    runtime=20:00:00
                elif [[ $mode -eq 16 ]]; then
                    runtime=26:00:00
                elif [[ $mode -eq 64 ]]; then
                    runtime=50:00:00
                fi
            fi
            echo "model: ${model} difficulty: ${difficulty} mode: ${mode} seed: ${seed} runtime: ${runtime}"
            sbatch --job-name=${model}-${mode}-${difficulty} --output=slurm/${model}_output_${difficulty}_${mode}_${seed}.txt --error=slurm/${model}_error_${difficulty}_${mode}_${seed}.txt --time=${runtime} 00-run_main.sh --difficulty ${difficulty} --num_modes ${mode} --seed ${seed} --model ${model}
        done
    done
done

# ==== MLP ====
# sbatch --output=slurm/mlp_output_e4.txt --error=slurm/mlp_error_e4.txt 00-run_main.sh --difficulty easy --num_modes 4 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_me4.txt --error=slurm/mlp_error_me4.txt 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_mh4.txt --error=slurm/mlp_error_mh4.txt 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_h4.txt --error=slurm/mlp_error_h4.txt 00-run_main.sh --difficulty hard --num_modes 4 --seed 1 --model mlp

# sbatch --output=slurm/mlp_output_e16.txt --error=slurm/mlp_error_e16.txt 00-run_main.sh --difficulty easy --num_modes 16 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_me16.txt --error=slurm/mlp_error_me16.txt 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_mh16.txt --error=slurm/mlp_error_mh16.txt 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_h16.txt --error=slurm/mlp_error_h16.txt 00-run_main.sh --difficulty hard --num_modes 16 --seed 1 --model mlp

# sbatch --output=slurm/mlp_output_e64.txt --error=slurm/mlp_error_e64.txt 00-run_main.sh --difficulty easy --num_modes 64 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_me64.txt --error=slurm/mlp_error_me64.txt 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_mh64.txt --error=slurm/mlp_error_mh64.txt 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 1 --model mlp
# sbatch --output=slurm/mlp_output_h64.txt --error=slurm/mlp_error_h64.txt 00-run_main.sh --difficulty hard --num_modes 64 --seed 1 --model mlp

# # TODO:

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 42 --model mlp

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 42 --model mlp

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 42 --model mlp
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 42 --model mlp

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 99 --model mlp

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 99 --model mlp

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 99 --model mlp
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 99 --model mlp

# # ==== LSTM ====
# sbatch --output=slurm/lstm_output_e4.txt --error=slurm/lstm_error_e4.txt 00-run_main.sh --difficulty easy --num_modes 4 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_me4.txt --error=slurm/lstm_error_me4.txt 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_mh4.txt --error=slurm/lstm_error_mh4.txt 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_h4.txt --error=slurm/lstm_error_h4.txt 00-run_main.sh --difficulty hard --num_modes 4 --seed 1 --model lstm

# sbatch --output=slurm/lstm_output_e16.txt --error=slurm/lstm_error_e16.txt 00-run_main.sh --difficulty easy --num_modes 16 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_me16.txt --error=slurm/lstm_error_me16.txt 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_mh16.txt --error=slurm/lstm_error_mh16.txt 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_h16.txt --error=slurm/lstm_error_h16.txt 00-run_main.sh --difficulty hard --num_modes 16 --seed 1 --model lstm

# sbatch --output=slurm/lstm_output_e64.txt --error=slurm/lstm_error_e64.txt 00-run_main.sh --difficulty easy --num_modes 64 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_me64.txt --error=slurm/lstm_error_me64.txt 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_mh64.txt --error=slurm/lstm_error_mh64.txt 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 1 --model lstm
# sbatch --output=slurm/lstm_output_h64.txt --error=slurm/lstm_error_h64.txt 00-run_main.sh --difficulty hard --num_modes 64 --seed 1 --model lstm

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 42 --model lstm

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 42 --model lstm

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 42 --model lstm
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 42 --model lstm

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 99 --model lstm

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 99 --model lstm

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 99 --model lstm
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 99 --model lstm

# # ==== CNN ====
# sbatch --output=slurm/cnn_output_e4.txt --error=slurm/cnn_error_e4.txt 00-run_main.sh --difficulty easy --num_modes 4 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_me4.txt --error=slurm/cnn_error_me4.txt 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_mh4.txt --error=slurm/cnn_error_mh4.txt 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_h4.txt --error=slurm/cnn_error_h4.txt 00-run_main.sh --difficulty hard --num_modes 4 --seed 1 --model cnn

# sbatch --output=slurm/cnn_output_e16.txt --error=slurm/cnn_error_e16.txt 00-run_main.sh --difficulty easy --num_modes 16 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_me16.txt --error=slurm/cnn_error_me16.txt 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_mh16.txt --error=slurm/cnn_error_mh16.txt 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_h16.txt --error=slurm/cnn_error_h16.txt 00-run_main.sh --difficulty hard --num_modes 16 --seed 1 --model cnn

# sbatch --output=slurm/cnn_output_e64.txt --error=slurm/cnn_error_e64.txt 00-run_main.sh --difficulty easy --num_modes 64 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_me64.txt --error=slurm/cnn_error_me64.txt 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_mh64.txt --error=slurm/cnn_error_mh64.txt 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 1 --model cnn
# sbatch --output=slurm/cnn_output_h64.txt --error=slurm/cnn_error_h64.txt 00-run_main.sh --difficulty hard --num_modes 64 --seed 1 --model cnn

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 42 --model cnn

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 42 --model cnn

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 42 --model cnn
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 42 --model cnn

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 99 --model cnn

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 99 --model cnn

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 99 --model cnn
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 99 --model cnn

# # ==== VAE ====
# sbatch --output=slurm/vae_output_e4.txt --error=slurm/vae_error_e4.txt 00-run_main.sh --difficulty easy --num_modes 4 --seed 1 --model vae
# sbatch --output=slurm/vae_output_me4.txt --error=slurm/vae_error_me4.txt 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model vae
# sbatch --output=slurm/vae_output_mh4.txt --error=slurm/vae_error_mh4.txt 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 1 --model vae
# sbatch --output=slurm/vae_output_h4.txt --error=slurm/vae_error_h4.txt 00-run_main.sh --difficulty hard --num_modes 4 --seed 1 --model vae

# sbatch --output=slurm/vae_output_e16.txt --error=slurm/vae_error_e16.txt 00-run_main.sh --difficulty easy --num_modes 16 --seed 1 --model vae
# sbatch --output=slurm/vae_output_me16.txt --error=slurm/vae_error_me16.txt 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 1 --model vae
# sbatch --output=slurm/vae_output_mh16.txt --error=slurm/vae_error_mh16.txt 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 1 --model vae
# sbatch --output=slurm/vae_output_h16.txt --error=slurm/vae_error_h16.txt 00-run_main.sh --difficulty hard --num_modes 16 --seed 1 --model vae

# sbatch --output=slurm/vae_output_e64.txt --error=slurm/vae_error_e64.txt 00-run_main.sh --difficulty easy --num_modes 64 --seed 1 --model vae
# sbatch --output=slurm/vae_output_me64.txt --error=slurm/vae_error_me64.txt 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 1 --model vae
# sbatch --output=slurm/vae_output_mh64.txt --error=slurm/vae_error_mh64.txt 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 1 --model vae
# sbatch --output=slurm/vae_output_h64.txt --error=slurm/vae_error_h64.txt 00-run_main.sh --difficulty hard --num_modes 64 --seed 1 --model vae

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 42 --model vae

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 42 --model vae

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 42 --model vae
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 42 --model vae

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 99 --model vae

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 99 --model vae

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 99 --model vae
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 99 --model vae

# # ==== SAVAR ====
# sbatch --output=slurm/picabu_output_e4.txt --error=slurm/picabu_error_e4.txt 00-run_main.sh --difficulty easy --num_modes 4 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_me4.txt --error=slurm/picabu_error_me4.txt 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_mh4.txt --error=slurm/picabu_error_mh4.txt 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_h4.txt --error=slurm/picabu_error_h4.txt 00-run_main.sh --difficulty hard --num_modes 4 --seed 1 --model picabu

# sbatch --output=slurm/picabu_output_e16.txt --error=slurm/picabu_error_e16.txt 00-run_main.sh --difficulty easy --num_modes 16 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_me16.txt --error=slurm/picabu_error_me16.txt 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_mh16.txt --error=slurm/picabu_error_mh16.txt 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_h16.txt --error=slurm/picabu_error_h16.txt 00-run_main.sh --difficulty hard --num_modes 16 --seed 1 --model picabu

# sbatch --output=slurm/picabu_output_e64.txt --error=slurm/picabu_error_e64.txt 00-run_main.sh --difficulty easy --num_modes 64 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_me64.txt --error=slurm/picabu_error_me64.txt 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_mh64.txt --error=slurm/picabu_error_mh64.txt 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 1 --model picabu
# sbatch --output=slurm/picabu_output_h64.txt --error=slurm/picabu_error_h64.txt 00-run_main.sh --difficulty hard --num_modes 64 --seed 1 --model picabu

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 42 --model picabu

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 42 --model picabu

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 42 --model picabu
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 42 --model picabu

# # sbatch 00-run_main.sh --difficulty easy --num_modes 4 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty hard --num_modes 4 --seed 99 --model picabu

# # sbatch 00-run_main.sh --difficulty easy --num_modes 16 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 16 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 16 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty hard --num_modes 16 --seed 99 --model picabu

# # sbatch 00-run_main.sh --difficulty easy --num_modes 64 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty med_easy --num_modes 64 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty med_hard --num_modes 64 --seed 99 --model picabu
# # sbatch 00-run_main.sh --difficulty hard --num_modes 64 --seed 99 --model picabu