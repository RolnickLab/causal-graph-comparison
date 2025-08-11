import json
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from climatem.model.train_model import TrainingLatent
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.model.metrics import edge_errors, mcc_latent, precision_recall, shd, w_mae
import torch
import time
import numpy as np

from causal_graph_comparison import MODELS_DIR, PLATFORM

torch.set_warn_always(False)


if PLATFORM == "cluster":
    cpu = False    
else:
    cpu = True

# accelerator = Accelerator(log_with="wandb", cpu=cpu)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

def train_picabu(
    datamodule,
    experiment_params,
    data_params,
    gt_params,
    train_params,
    model_params,
    optim_params,
    plot_params,
    savar_params,
    wandb,
    trained_model = None, # if None, train picabu on savar, otherwise train on model-generated data
    trained_model_params = None,
):
    # to run picabu as VAE, value should be 1e-8
    vae_mode = optim_params.ortho_mu_init < 1 

    # add wandb to accelerator trackers
    accelerator.trackers = [wandb]

    # Set the name of the run based on what picabu is trained on
    if trained_model is not None:
        trained_model_name = f"picabu_{trained_model.name}"
    elif vae_mode:
        trained_model_name = "vae"
    else:
        trained_model_name = "picabu_savar"

    save_name = f"modes_{experiment_params.d_z}-diff_{savar_params.difficulty}-seed_{experiment_params.random_seed}"
    name = f"{trained_model_name}-{save_name}"
    exp_path = Path(experiment_params.exp_path) / name
    exp_path.mkdir(exist_ok=True)

    # check if model already exists
    if (exp_path / "model-final.pth").exists():
        print(f"=== SKIPPING TRAINING: Model already exists at {exp_path / 'model-final.pth'}")
        return

    t0 = time.time()
    
    d = len(data_params.in_var_ids) # number of datasets, in this case = 1
    
    # set the picabumodel
    picabu_model = LatentTSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        num_input=experiment_params.tau,
        num_output=experiment_params.future_timesteps, 
        num_layers_mixing=model_params.num_layers_mixing,
        num_hidden_mixing=model_params.num_hidden_mixing,
        position_embedding_dim=model_params.position_embedding_dim,
        reduce_encoding_pos_dim=model_params.reduce_encoding_pos_dim,
        coeff_kl=optim_params.coeff_kl,
        d=d, # number of datasets
        # Here, everything hardcoded to gaussian because GEV leads to Nan... TBD
        distr_z0="gaussian",
        distr_encoder="gaussian",
        distr_transition="gaussian",
        distr_decoder="gaussian",
        d_x=experiment_params.d_x,
        d_z=experiment_params.d_z,
        tau=experiment_params.tau,
        instantaneous=model_params.instantaneous,
        nonlinear_dynamics=model_params.nonlinear_dynamics,
        nonlinear_mixing=model_params.nonlinear_mixing,
        hard_gumbel=model_params.hard_gumbel,
        no_gt=gt_params.no_gt,
        debug_gt_graph=gt_params.debug_gt_graph,
        debug_gt_z=gt_params.debug_gt_z,
        debug_gt_w=gt_params.debug_gt_w,
        tied_w=model_params.tied_w,
        fixed=model_params.fixed,
        fixed_output_fraction=model_params.fixed_output_fraction,
        vae_mode=vae_mode,
    )

    # Set wandb logging
    wandb.watch(picabu_model, log="all")

    save_path = exp_path / "training_results"
    save_path.mkdir(exist_ok=True)
    plots_path = exp_path / "plots"
    plots_path.mkdir(exist_ok=True)

    # Save hyperparameters to file
    hp = {}
    hp["exp_params"] = experiment_params.__dict__
    hp["data_params"] = data_params.__dict__
    hp["gt_params"] = gt_params.__dict__
    hp["train_params"] = train_params.__dict__
    hp["model_params"] = model_params.__dict__
    hp["optim_params"] = optim_params.__dict__
    hp["trained_model"] = trained_model_params if trained_model_params is not None else None

    with open(exp_path / "params.json", "w") as file:
        json.dump(hp, file, indent=4)

    best_metrics = {"recons": 0, "kl": 0, "mcc": 0, "elbo": 0}


    wandbname = f"picabu-{trained_model.name if trained_model is not None else ''}{'vae' if vae_mode else ''}"
    # train, always with the latent version
    trainer = TrainingLatent(
        picabu_model,
        datamodule,
        experiment_params,
        gt_params,
        model_params,
        train_params,
        optim_params,
        plot_params,
        save_path,
        plots_path,
        best_metrics,
        d, # number of datasets
        accelerator,
        trained_model=trained_model,
        trained_model_type=trained_model.name if trained_model is not None else None,
        wandbname=wandbname,
        profiler=False,
    )

    # TODO: delete
    # Where is the model at this point?
    print("Where is my model?", next(trainer.model.parameters()).device)

    valid_loss = trainer.train_with_QPM()

    # assert that trainer.model is in eval mode
    if trainer.model.training:
        print("Model is in train mode")
    else:
        print("Model is in eval mode")

    # finally, save the model
    torch.save(trainer.model.state_dict(), exp_path / "model-final.pth")
    
    if vae_mode:
        torch.save(trainer.model.state_dict(), MODELS_DIR / f"{name}.pth")
    