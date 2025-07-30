import argparse
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from causal_graph_comparison.mlp import Net
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from causal_graph_comparison.utils import get_json_config
from causal_graph_comparison.test_dataloader import SineTestDataset

# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
from causal_graph_comparison import *
from datetime import datetime
from causal_graph_comparison.utils import flatten_data_target

# Training settings
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.0001
GAMMA = 0.7
NO_ACCEL = False
DRY_RUN = False
SEED = 1
LOG_INTERVAL = 10
SAVE_MODEL = True
FUTURE_TIMESTEPS = 1
LONGITUDE = 40
LATITUDE = 40
TAU = 5
LAYERS = [1600, 800, 1600]  # try with 4
INPUT_SIZE = LATITUDE * LONGITUDE * TAU
OUTPUT_SIZE = LATITUDE * LONGITUDE * FUTURE_TIMESTEPS
DIFFICULTY = "med_easy"
NUM_MODES = 4
MODEL_TYPE = "mlp"
DATASET_TYPE = "savar"
BEST = False
best_name = "best-" if BEST else ""

diff_mapping = {
    "easy": "e",
    "med_easy": "me",
    "med_hard": "mh",
    "hard": "h",
}

# SINE DATASET TESTING
# LAYERS = [30, 20, 10]
# INPUT_SIZE = 30
# OUTPUT_SIZE = 1

wandb.init(
    project="climatem",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "tau": TAU,
        "future_timesteps": FUTURE_TIMESTEPS,
        "model": "mlp",
        "layers": LAYERS,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "seed": SEED,
        "gamma": GAMMA,
    },
)

def train(log_interval: int, dry_run: bool, model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int):
    """
    Training loop for mlp
    data shape: torch.Size([128, 5, 1, 1600])
    target shape: torch.Size([128, 1, 1, 1600])
    args:
        log_interval: interval to log training loss
        dry_run: if True, only run for 1 batch
        model: mlp model
        device: device to train on
        train_loader: dataloader for training data
        optimizer: optimizer to use
        epoch: current epoch
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # flatten the data and target
        # data, target = flatten_data_target(data, target, INPUT_SIZE, OUTPUT_SIZE)
        # print("==========")
        # print("batch idx: ", batch_idx)
        # print("")
        # print(f"data shape from train: {data.shape}")
        # print(f"target shape from train: {target.shape}")
        # print("--------------")

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        wandb.log({"loss_train": loss.item() / len(data)})
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break


def test(model: nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader):
    """
    Testing loop for mlp
    args:
        model: mlp model
        device: device to test on
        test_loader: dataloader for testing data
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:

            # flatten the data and target
            # data, target = flatten_data_target(data, target, INPUT_SIZE, OUTPUT_SIZE)

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}\n")
    wandb.log({"loss_valid": test_loss})

    return test_loss

def train_model(model, ds_train, ds_test, model_path):
    # TODO: run train_model from trainer.py
    pass

config_dict = get_json_config("mlp_config.json")
# TODO get config params


use_accel = not NO_ACCEL and torch.cuda.is_available()

torch.manual_seed(SEED)

if use_accel:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_kwargs = {"batch_size": BATCH_SIZE}
test_kwargs = {"batch_size": TEST_BATCH_SIZE}
if use_accel:
    accel_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)

# =======  Test dataset =======

# print ("==== USING SINE TEST DATASET ====")
# dataset_train = SineTestDataset(num_samples=1000)
# dataset_test = SineTestDataset(num_samples=1000, test=True)

# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False)

# =======  SAVAR dataset =======

print("==== USING SAVAR DATASET ====")

dl = CausalClimateDataModule(
    # Required parameters for ClimateDataModule
    d_z=NUM_MODES,
    in_var_ids=["savar"],
    out_var_ids=["savar"],  # Same as input for SAVAR
    train_years="2015-2100",
    train_historical_years="1950-2014",
    test_years="2015-2100",  # Same as train for SAVAR
    val_split=0.1,  # 10% validation split
    seq_to_seq=True,
    channels_last=False,
    train_scenarios=["savar"],
    test_scenarios=["savar"],
    train_models="savar",
    batch_size=BATCH_SIZE,
    eval_batch_size=128,
    num_workers=0,
    pin_memory=False,
    load_train_into_mem=True,
    load_test_into_mem=True,
    verbose=True,
    seed=SEED,
    seq_len=12,
    data_dir="",  # Not used for SAVAR
    output_save_dir=f"{SCRATCH_DIR}/data/SAVAR_DATA_TEST",
    num_ensembles=1,
    lon=LONGITUDE,
    lat=LATITUDE,
    num_levels=1,
    global_normalization=True,
    seasonality_removal=False,
    reload_climate_set_data=True,
    # Required parameters for CausalClimateDataModule
    tau=TAU,
    future_timesteps=FUTURE_TIMESTEPS,
    num_months_aggregated=1,
    train_val_interval_length=100,
    # SAVAR specific parameters
    time_len=10000,
    comp_size=10,
    noise_val=0.2,
    n_per_col=2,
    difficulty="med_easy",
    seasonality=False,
)

dl.setup()

train_dataset = dl._data_train
test_dataset = dl._data_val

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================

# How they do it in picabu

# from accelerate import Accelerator
# from accelerate.utils import DistributedDataParallelKwargs

# kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

# train_loader = dl.train_dataloader(accelerator)
# test_loader = dl.val_dataloader()

# x, y = next(test_loader)
# x = torch.nan_to_num(x)
# y = torch.nan_to_num(y)
# y = y[:, 0]

# x = x.to(device)
# y = y.to(device)

# ============ RUN TRAINING ============

model = Net(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, layers=LAYERS).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

# basline test with untrained model
best_test_loss = test(model, device, test_loader)

save_name = f"{best_name}{MODEL_TYPE}-{DATASET_TYPE}-modes_{NUM_MODES}-diff_{diff_mapping[DIFFICULTY]}-seed_{SEED}"

for epoch in range(1, EPOCHS + 1):
    train(LOG_INTERVAL, DRY_RUN, model, device, train_loader, optimizer, epoch)
    test_loss = test(model, device, test_loader)
    scheduler.step()

    if SAVE_MODEL:

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model, f"{MODELS_DIR}/{save_name}.pt")
        torch.save(model, f"{MODELS_DIR}/{save_name}.pt")
