import numpy as np
from pathlib import Path

from causal_graph_comparison import *
from climatem.synthetic_data.graph_evaluation_ilija import *
from climatem.plotting.plot_model_output import Plotter

class PicabuResultsPaths:
    def __init__(self, exp_params, savar_params, model_name):
        
        self.model_name = model_name
        self.path_name = f"modes_{exp_params.d_z}-difficulty_{savar_params.difficulty}-seed_{exp_params.random_seed}"
        self.results_path = Path(f"{SCRATCH_DIR}/results/SAVAR_DATA_TEST/{self.model_name}-{self.path_name}")

        print(f"Loading learned temporal graph & other results from: {self.results_path} for model: {self.model_name}")
        self.plots_path = self.results_path / Path("plots")
        self.training_results_path = self.results_path / Path("training_results")

        self.graph_path = self.plots_path / Path("graphs.npy")
        self.graph = np.load(self.graph_path)
        self.w_decoder_path = self.plots_path / Path("w_decoder.npy")
        self.w_decoder = np.load(self.w_decoder_path)
        self.w_encoder_path = self.plots_path / Path("w_encoder.npy")
        self.w_encoder = np.load(self.w_encoder_path)

        # gt from savar
        self.savar_path = SCRATCH_DIR / Path("data/SAVAR_DATA_TEST") 
        self.modes_gt_path = self.savar_path / Path(f"{self.path_name}_mode_weights.npy")
        self.modes_gt = np.load(self.modes_gt_path)

def permute_graph(datamodule, exp_params, savar_params, model_name):

    results = PicabuResultsPaths(exp_params, savar_params, model_name)

    # load results
    learned_graph = results.graph
    inferred_modes = results.w_decoder

    # load params
    lat = datamodule.hparams.lat
    lon = datamodule.hparams.lon
    tau = datamodule.tau

    # load savar gt from datamodule
    print(f"Loading savar gt from SAVAR datamodule")
    savar_gt = datamodule.savar_gt_adj
    modes_gt = datamodule.savar_gt_modes_weights

    print("savar_gt: ", savar_gt.shape)
    print("modes_gt from datamodule: ", modes_gt.shape)

    # if loading from dataloader still doesnt work after reloading the datamodule, use this instead
    # modes_gt = results.modes_gt
    # print("modes_gt from numpy: ", modes_gt.shape)

    plotter = Plotter()

    # Plot learned graph vs savar gt before permutation
    plotter.plot_adjacency_matrix(
        mat1=learned_graph,
        mat2=savar_gt[::-1],
        path=results.results_path,
        name_suffix="transition",
        no_gt=False,
        iteration=1,
        plot_through_time=True,
    )

    # TODO: remember hardcoded permutation function for 100 mondes
    # Permute learned graph using CDSD modes
    permuted_temporal_matrix = np.array(
        load_and_permute_all_matrices(inferred_modes, modes_gt, learned_graph, savar_gt, lat, lon, tau)
    )

    print("permuted_temporal_matrix shape: ", permuted_temporal_matrix.shape)
    print("permuted_temporal_matrix: ", permuted_temporal_matrix)

    # Plot permuted graph vs savar gt after permutation
    plotter.plot_adjacency_matrix(
        mat1=permuted_temporal_matrix,
        mat2=savar_gt[::-1],
        path=results.results_path,
        name_suffix="transition_permuted",
        no_gt=False,
        iteration=2,
        plot_through_time=True,
    )

    np.save(results.results_path / "permuted_learned_temporal_graph.npy", permuted_temporal_matrix)

    return permuted_temporal_matrix
