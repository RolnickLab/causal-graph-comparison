# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_easy-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_easy-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_easy-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_easy-seed_1/

# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_easy-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_easy-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_easy-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_easy-seed_1/

# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_hard-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_hard-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_hard-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_med_hard-seed_1/

# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_hard-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_hard-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_hard-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_4-diff_hard-seed_1/

# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_easy-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_easy-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_easy-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_easy-seed_1/

# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_easy-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_easy-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_easy-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_easy-seed_1/

# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_hard-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_hard-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_hard-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_med_hard-seed_1/

# rsync -ratulvzP --info=progress2 "mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_hard-seed_1/adjacency_transition_*" ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_hard-seed_1/
# rsync -ratulvzP --info=progress2 mila:~/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_hard-seed_1/plots ~/dev/mila/scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_16-diff_hard-seed_1/

# experiment_name = "picabu_savar-modes_4-diff_easy-seed_1-sparsitymuinit_0.001-sparsitymumultfactor_1.5"
# experiment_parts = experiment_name.split("-")
# data_name = "-".join(experiment_parts[1:4])
# print(data_name)

# --> test order of extract_adjacency_matrix function (does it output graph in forward direction?)
# --> in savar_dataset.py
np.array(extract_adjacency_matrix(self.links_coeffs, self.n_per_col**2, tau))