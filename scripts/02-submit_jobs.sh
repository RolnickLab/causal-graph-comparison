seed=1
for mode in 4 16; do
    for model in mlp lstm cnn vae picabu; do
        for difficulty in easy med_easy med_hard hard; do
            runtime=16:00:00
            if [[ $mode -eq 4 ]]; then
                runtime=12:00:00
            elif [[ $mode -eq 16 ]]; then
                runtime=16:00:00
            fi
            echo "model: ${model} difficulty: ${difficulty} mode: ${mode} seed: ${seed} runtime: ${runtime}"
            sbatch --job-name=${model}-${mode}-${difficulty} --output=slurm/${model}_output_${difficulty}_${mode}_${seed}.txt --error=slurm/${model}_error_${difficulty}_${mode}_${seed}.txt --time=${runtime} 00-run_main.sh --difficulty ${difficulty} --num_modes ${mode} --seed ${seed} --model ${model}
        done
    done
done