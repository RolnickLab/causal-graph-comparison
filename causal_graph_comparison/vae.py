from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison import MODELS_DIR


def train_vae(datamodule, experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params, wandb, trained_model_params):

    # run the linear vae

    vae_optim_params = optim_params.copy()

    vae_optim_params.ortho_mu_init = trained_model_params["vae"]["optim_params"]["ortho_mu_init"]
    vae_optim_params.ortho_mu_mult_factor = trained_model_params["vae"]["optim_params"]["ortho_mu_mult_factor"]
    vae_optim_params.ortho_h_threshold = trained_model_params["vae"]["optim_params"]["ortho_h_threshold"]

    vae_optim_params.sparsity_mu_init = trained_model_params["vae"]["optim_params"]["sparsity_mu_init"]
    vae_optim_params.sparsity_mu_mult_factor = trained_model_params["vae"]["optim_params"]["sparsity_mu_mult_factor"]
    vae_optim_params.sparsity_h_threshold = trained_model_params["vae"]["optim_params"]["sparsity_h_threshold"]

    vae_optim_params.acyclic_mu_init = trained_model_params["vae"]["optim_params"]["acyclic_mu_init"]
    vae_optim_params.acyclic_mu_mult_factor = trained_model_params["vae"]["optim_params"]["acyclic_mu_mult_factor"]
    vae_optim_params.acyclic_h_threshold = trained_model_params["vae"]["optim_params"]["acyclic_h_threshold"]

    train_picabu(datamodule, experiment_params, data_params, gt_params, train_params, model_params, vae_optim_params, plot_params, savar_params, wandb, trained_model=None, trained_model_params=trained_model_params, linear=True)
    model_path = f"{MODELS_DIR}/vae-modes_{experiment_params.d_z}-diff_{savar_params.difficulty}-seed_{experiment_params.random_seed}-linear.pth"

    return model_path