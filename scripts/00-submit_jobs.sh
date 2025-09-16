# === run chris
seed=1
for mode in 4 16; do
    for model in mlp lstm cnn; do
        for difficulty in easy hard; do
            runtime=1:00:00
            # if [[ $mode -eq 4 ]]; then
            #     runtime=12:00:00
            # elif [[ $mode -eq 16 ]]; then
            #     runtime=14:30:00
            # elif [[ $mode -eq 36 ]]; then
            #     runtime=40:00:00
            # fi
            if [[ $mode -eq 4 ]] && [[ $difficulty == "med_easy" ]]; then
                echo "Skipping med_easy for 4 modes"
                continue
            elif [[ $mode -eq 4 ]] && [[ $difficulty == "med_hard" ]]; then
                echo "Skipping med_hard for 4 modes"
                continue
            fi
            echo "model: ${model} difficulty: ${difficulty} mode: ${mode} seed: ${seed} runtime: ${runtime}"
            sbatch --job-name=L${model}-${mode}-${difficulty} --output=slurm/L${model}_output_${difficulty}_${mode}_${seed}.txt --error=slurm/L${model}_error_${difficulty}_${mode}_${seed}.txt --time=${runtime} 00-run_main.sh --difficulty ${difficulty} --num_modes ${mode} --seed ${seed} --model ${model}
        done
    done
done

# sbatch --job-name=Lvae-16-easy --output=slurm/Lvae_output_easy_16_1.txt --error=slurm/Lvae_error_easy_16_1.txt --time=20:00:00 00-run_main.sh --difficulty easy --num_modes 16 --seed 1 --model vae

# sbatch --job-name=NLvae-4-med_easy --output=slurm/NLvae_output_med_easy_4_1.txt --error=slurm/NLvae_error_med_easy_4_1.txt --time=12:00:00 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model vae
# sbatch --job-name=NLvae-4-med_hard --output=slurm/NLvae_output_med_hard_4_1.txt --error=slurm/NLvae_error_med_hard_4_1.txt --time=12:00:00 00-run_main.sh --difficulty med_hard --num_modes 4 --seed 1 --model vae
# sbatch --job-name=picabu-4-med_easy --output=slurm/picabu_output_med_easy_4_1.txt --error=slurm/picabu_error_med_easy_4_1.txt --time=8:00:00 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model picabu
# sbatch --job-name=Lpicabu-4-med_easy --output=slurm/Lpicabu_output_med_easy_4_1.txt --error=slurm/Lpicabu_error_med_easy_4_1.txt --time=8:00:00 00-run_main.sh --difficulty med_easy --num_modes 4 --seed 1 --model picabu

# sbatch --job-name=savar-64mh_mui01_muf15 --output=slurm/savar_output_64mh_mui01_muf15.txt --error=slurm/savar_error_64mh_mui01_muf15.txt --time=40:00:00 00-run_main.sh --difficulty med_hard --num_modes 64 --model picabu --sparsity_mu_init 0.01 --sparsity_mu_mult_factor 1.5
# sbatch --job-name=savar-64mh_mui001_muf2 --output=slurm/savar_output_64mh_mui001_muf2.txt --error=slurm/savar_error_64mh_mui001_muf2.txt --time=40:00:00 00-run_main.sh --difficulty med_hard --num_modes 64 --model picabu --sparsity_mu_init 0.001 --sparsity_mu_mult_factor 2
# sbatch --job-name=savar-64mh_mui01_muf2 --output=slurm/savar_output_64mh_mui01_muf2.txt --error=slurm/savar_error_64mh_mui01_muf2.txt --time=40:00:00 00-run_main.sh --difficulty med_hard --num_modes 64 --model picabu --sparsity_mu_init 0.01 --sparsity_mu_mult_factor 2

# sbatch --job-name=savar-16mh_mui01_muf15 --output=slurm/savar_output_16mh_mui01_muf15.txt --error=slurm/savar_error_16mh_mui01_muf15.txt --time=20:00:00 00-run_main.sh --difficulty med_hard --num_modes 16 --model picabu --sparsity_mu_init 0.01 --sparsity_mu_mult_factor 1.5
# sbatch --job-name=savar-16mh_mui001_muf2 --output=slurm/savar_output_16mh_mui001_muf2.txt --error=slurm/savar_error_16mh_mui001_muf2.txt --time=20:00:00 00-run_main.sh --difficulty med_hard --num_modes 16 --model picabu --sparsity_mu_init 0.001 --sparsity_mu_mult_factor 2
# sbatch --job-name=savar-16mh_mui01_muf2 --output=slurm/savar_output_16mh_mui01_muf2.txt --error=slurm/savar_error_16mh_mui01_muf2.txt --time=20:00:00 00-run_main.sh --difficulty med_hard --num_modes 16 --model picabu --sparsity_mu_init 0.01 --sparsity_mu_mult_factor 2

# sbatch --job-name=vae-64-hard --output=slurm/vae_output_hard_64_1.txt --error=slurm/vae_error_hard_64_1.txt --time=55:00:00 00-run_main.sh --difficulty hard --num_modes 64 --seed 1 --model vae
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