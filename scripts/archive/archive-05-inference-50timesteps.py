import argparse
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from causal_graph_comparison.lstm import Lstm_sine
from causal_graph_comparison.test_dataloader import SineTestDataset
import matplotlib.pyplot as plt

# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
from causal_graph_comparison import *
from datetime import datetime
from causal_graph_comparison.utils import flatten_data_target
from causal_graph_comparison.mlp import Net
import glob

TIMESTAMP = "2025_07_10_00_04_54"

# Test settings
TEST_BATCH_SIZE = 1
NO_ACCEL = False
SEED = 1
FUTURE_TIMESTEPS = 1
ROLLOUT_TIMESTEPS = 50
LONGITUDE = 40
LATITUDE = 40
TAU = 5
LAYERS = [1600, 800, 1600]
INPUT_SIZE = LATITUDE * LONGITUDE * TAU
OUTPUT_SIZE = LATITUDE * LONGITUDE * FUTURE_TIMESTEPS


def inference(model, device, test_loader):

    data_list = []
    target_list = []
    output_list = []

    model.eval()

    with torch.no_grad():
        # get first batch
        data, target = next(iter(test_loader))

        for x in data[0]:
            data_list.append(x.squeeze().cpu().numpy())
        data = data.to(device)

        # Flatten data for MLP sine data: (batch, seq_len, features) -> (batch, seq_len*features)
        # data = data.view(data.shape[0], -1)

        predictions = []
        h0 = None
        c0 = None
        for step in range(ROLLOUT_TIMESTEPS):
            # print("step: ", step)
            # ======== LSTM ========
            # drop 3rd dimension for savar data
            # data = data.squeeze(2)
            # output, h0, c0 = model(data, h0=h0, c0=c0)

            # ======== MLP ========
            output = model(data.view(data.shape[0], -1))

            output_list.append(output.squeeze().cpu().numpy())

            data = torch.roll(data, shifts=-1, dims=1)
            data[:, -1] = output

    for i, (data, target) in enumerate(test_loader):
        target_list.append(target.squeeze().cpu().numpy())
        if i == ROLLOUT_TIMESTEPS - 1:
            break

    # Convert lists to arrays
    data_array = np.array(data_list)
    target_array = np.array(target_list)
    output_array = np.array(output_list)

    print("Data array shape: ", data_array.shape)
    print("Target array shape: ", target_array.shape)
    print("Output array shape: ", output_array.shape)

    np.savez(OUTPUTS_DIR / f"test_results-mlp-savar.npz", inputs=data_array, target=target_array, outputs=output_array)


use_accel = not NO_ACCEL and torch.cuda.is_available()

torch.manual_seed(SEED)

if use_accel:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_kwargs = {"batch_size": TEST_BATCH_SIZE}
if use_accel:
    accel_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    test_kwargs.update(accel_kwargs)

# =======  SAVAR dataset =======

dl = CausalClimateDataModule(
    # Required parameters for ClimateDataModule
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
    eval_batch_size=64,
    num_workers=0,
    pin_memory=False,
    load_train_into_mem=True,
    load_test_into_mem=True,
    verbose=True,
    seed=42,
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

test_dataset = dl._data_val

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# =======  Test dataset =======

# dataset = SineTestDataset(num_samples=1000, test=True)

# test_loader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# =======  Run inference on model =======

# find model by glob

# model_name = "savar_lstm-best-"
# model_name = "savar_lstm-2025_"
model_name = "savar_mlp-2025_"
# model_name = "savar_mlp-best-"
model_path = glob.glob(f"{MODELS_DIR}/{model_name}*.pt")[1]
print("model_path: ", model_path)

model = torch.load(model_path, map_location=device)

inference(model, device, test_loader)
