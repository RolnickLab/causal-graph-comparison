import copy
from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison import MODELS_DIR


def train_vae(datamodule, experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params, wandb, trained_model_params):
    """Train a VAE model, which is PICABU with constraints turned off.

    Args:
        datamodule: DataModule containing training data
        experiment_params: Parameters for experiment setup
        data_params: Parameters for data loading and processing
        gt_params: Ground truth parameters
        train_params: Training parameters
        model_params: Model architecture parameters
        optim_params: Optimization parameters
        plot_params: Plotting parameters
        savar_params: SAVAR model parameters
        wandb: Weights & Biases logger
        trained_model_params: Parameters from pretrained model

    Returns:
        Path to saved model file
    """

    vae_optim_params = copy.deepcopy(optim_params)

    vae_optim_params.ortho_mu_init = trained_model_params["vae"]["optim_params"]["ortho_mu_init"]
    vae_optim_params.ortho_mu_mult_factor = trained_model_params["vae"]["optim_params"]["ortho_mu_mult_factor"]
    vae_optim_params.ortho_h_threshold = trained_model_params["vae"]["optim_params"]["ortho_h_threshold"]

    vae_optim_params.sparsity_mu_init = trained_model_params["vae"]["optim_params"]["sparsity_mu_init"]
    vae_optim_params.sparsity_mu_mult_factor = trained_model_params["vae"]["optim_params"]["sparsity_mu_mult_factor"]
    vae_optim_params.sparsity_h_threshold = trained_model_params["vae"]["optim_params"]["sparsity_h_threshold"]

    vae_optim_params.acyclic_mu_init = trained_model_params["vae"]["optim_params"]["acyclic_mu_init"]
    vae_optim_params.acyclic_mu_mult_factor = trained_model_params["vae"]["optim_params"]["acyclic_mu_mult_factor"]
    vae_optim_params.acyclic_h_threshold = trained_model_params["vae"]["optim_params"]["acyclic_h_threshold"]

    train_picabu(datamodule, experiment_params, data_params, gt_params, train_params, model_params, vae_optim_params, plot_params, savar_params, wandb, trained_model=None, trained_model_params=trained_model_params, linearity="linear")
    model_path = f"{MODELS_DIR}/vae-modes_{experiment_params.d_z}-diff_{savar_params.difficulty}-seed_{experiment_params.random_seed}.pth"

    return model_path