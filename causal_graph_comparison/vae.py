from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison import MODELS_DIR


def train_vae(datamodule, experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params, wandb, trained_model_params):

    optim_params.ortho_mu_init = trained_model_params["vae"]["optim_params"]["ortho_mu_init"]
    optim_params.ortho_mu_mult_factor = trained_model_params["vae"]["optim_params"]["ortho_mu_mult_factor"]
    optim_params.ortho_h_threshold = trained_model_params["vae"]["optim_params"]["ortho_h_threshold"]

    optim_params.sparsity_mu_init = trained_model_params["vae"]["optim_params"]["sparsity_mu_init"]
    optim_params.sparsity_mu_mult_factor = trained_model_params["vae"]["optim_params"]["sparsity_mu_mult_factor"]
    optim_params.sparsity_h_threshold = trained_model_params["vae"]["optim_params"]["sparsity_h_threshold"]

    optim_params.acyclic_mu_init = trained_model_params["vae"]["optim_params"]["acyclic_mu_init"]
    optim_params.acyclic_mu_mult_factor = trained_model_params["vae"]["optim_params"]["acyclic_mu_mult_factor"]
    optim_params.acyclic_h_threshold = trained_model_params["vae"]["optim_params"]["acyclic_h_threshold"]

    train_picabu(datamodule, experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params, wandb, trained_model=None, trained_model_params=trained_model_params)
    model_path = f"{MODELS_DIR}/vae-modes_{experiment_params.d_z}-diff_{savar_params.difficulty}-seed_{experiment_params.random_seed}.pth"

    return model_path