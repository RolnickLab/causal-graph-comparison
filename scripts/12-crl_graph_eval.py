import numpy as np
from pathlib import Path
from causal_graph_comparison import *
import glob
from climatem.synthetic_data.graph_evaluation_ilija import *
from pathlib import Path

from importlib import reload
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from climatem.plotting.plot_model_output import Plotter

# Training settings
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
GAMMA = 0.9
NO_ACCEL = False
DRY_RUN = False
SEED = 1
LOG_INTERVAL = 10
SAVE_MODEL = True
FUTURE_TIMESTEPS = 1
LONGITUDE = 40
LATITUDE = 40
TAU = 5
DIFFICULTY = "med_easy"
NUM_MODES = 4
BEST = False
MODEL_TYPE = "cnn"
DATASET_TYPE = "savar"

best_name = "best-" if BEST else ""

diff_mapping = {
    "easy": "e",
    "med_easy": "me",
    "med_hard": "mh",
    "hard": "h",
}

model = "cnn"  # vae, mlp, lstm
base_path = SCRATCH_DIR / "results" / "SAVAR_DATA_TEST" / "learn_causal_graphs"
model_path = glob.glob(str(base_path / f"{model}-*"))
results_path = Path(model_path[0]) / "plots"
print("results path: ", results_path)
graphs_path = Path(results_path) / "graphs.npy"
graphs_file = glob.glob(str(graphs_path))[0]
graphs = np.load(graphs_file)

print("using graphs from: ", graphs_file)

print(graphs.shape)
print(graphs[0])

# Load parameters from npy file
savar_folder = SCRATCH_DIR / "data" / "SAVAR_DATA_TEST"
savar_fname = "modes_4_tl_10000_isforced_False_difficulty_med_easy_noisestrength_0.2_seasonality_False_overlap_False"
params_file = savar_folder / f"{savar_fname}_parameters.npy"
params = np.load(params_file, allow_pickle=True).item()
links_coeffs = params["links_coeffs"]
n_modes_gt = params["N"]
lat = params["nx"]
lon = params["ny"]
tau = 5
modes_gt = np.load(savar_folder / f"{savar_fname}_mode_weights.npy")
print("modes_gt shape: ", modes_gt.shape)
print("modes_gt: ", modes_gt[0])

gt_adj_list = extract_adjacency_matrix(links_coeffs, n_modes_gt, tau)

print(gt_adj_list.shape)
print(gt_adj_list[0])

# datamodule = CausalClimateDataModule(
#     # Required parameters for ClimateDataModule
#     in_var_ids=["savar"],
#     out_var_ids=["savar"],  # Same as input for SAVAR
#     train_years="2015-2100",
#     train_historical_years="1950-2014",
#     test_years="2015-2100",  # Same as train for SAVAR
#     val_split=0.1,  # 10% validation split
#     seq_to_seq=True,
#     channels_last=False,
#     train_scenarios=["savar"],
#     test_scenarios=["savar"],
#     train_models="savar",
#     eval_batch_size=64,
#     num_workers=0,
#     pin_memory=False,
#     load_train_into_mem=True,
#     load_test_into_mem=True,
#     verbose=True,
#     seed=42,
#     seq_len=12,
#     data_dir="",  # Not used for SAVAR
#     output_save_dir=f"{SCRATCH_DIR}/data/SAVAR_DATA_TEST",
#     num_ensembles=1,
#     lon=LONGITUDE,
#     lat=LATITUDE,
#     num_levels=1,
#     global_normalization=True,
#     seasonality_removal=False,
#     reload_climate_set_data=True,
#     # Required parameters for CausalClimateDataModule
#     tau=TAU,
#     future_timesteps=FUTURE_TIMESTEPS,
#     num_months_aggregated=1,
#     train_val_interval_length=100,
#     # SAVAR specific parameters
#     time_len=10000,
#     comp_size=10,
#     noise_val=0.2,
#     n_per_col=2,
#     difficulty="med_easy",
#     seasonality=False,
# )
#
# datamodule.setup()

# savar_gt = datamodule.savar_gt_adj

plotter = Plotter()

plotter.plot_adjacency_matrix(
    mat1=graphs,
    # Below savar dag
    # mat2=savar_gt,
    # mat2=gt_adj_list,
    mat2=gt_adj_list[::-1],
    path=results_path,
    name_suffix="transition",
    no_gt=False,
    iteration=1,
    plot_through_time=True,
)


# load CDSD results
cdsd_modes_inferred_path = results_path / "w_decoder.npy"
modes_inferred = np.load(cdsd_modes_inferred_path)

permuted_matrices = np.array(
    load_and_permute_all_matrices(modes_inferred, modes_gt, graphs, gt_adj_list, lat, lon, tau)
)

print("permuted_matrices shape: ", permuted_matrices.shape)
print("permuted_matrices: ", permuted_matrices)

plotter.plot_adjacency_matrix(
    mat1=permuted_matrices,
    # Below savar dag
    # mat2=savar_gt,
    # mat2=gt_adj_list,
    mat2=gt_adj_list[::-1],
    path=results_path,
    name_suffix="transition",
    no_gt=False,
    iteration=2,
    plot_through_time=True,
)
