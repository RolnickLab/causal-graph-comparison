import argparse
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
from causal_graph_comparison import CONFIGS_PATH, DATA_DIR, APP_ROOT, PROJECT_ROOT, SCRIPTS_DIR, OUTPUTS_DIR, SCRATCH_DIR, MODELS_DIR
from datetime import datetime
from causal_graph_comparison.utils import flatten_data_target

TIMESTAMP = "2025_07_01_15_04_50"

# Training settings
TEST_BATCH_SIZE = 1000
NO_ACCEL = False
SEED = 1
FUTURE_TIMESTEPS = 5
LONGITUDE = 40
LATITUDE = 40
TAU = 5
LAYERS = [1600, 800, 1600]
INPUT_SIZE = LATITUDE * LONGITUDE * TAU
OUTPUT_SIZE = LATITUDE * LONGITUDE * FUTURE_TIMESTEPS
SAVE_DIR = OUTPUTS_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)

class Net(nn.Module):
    def __init__(self, layers):
        super().__init__()

        activation = nn.LeakyReLU
        num_layers = len(layers)
        module_dict = OrderedDict()

        # Create model layer by layer
        module_dict["lin0"] = nn.Linear(INPUT_SIZE, layers[0])

        for layer in range(num_layers):
            in_features = layers[layer]
            out_features = layers[layer + 1] if layer < num_layers - 1 else OUTPUT_SIZE

            module_dict[f"nonlin{layer}"] = activation()
            module_dict[f"lin{layer+1}"] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)


    def forward(self, x):
        output = self.model(x)
        return output

def test(model, device, test_loader):

    data_list = []
    target_list = []
    output_list = []

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:

            # flatten the data and target
            data, target = flatten_data_target(data, target, INPUT_SIZE, OUTPUT_SIZE)

            data, target = data.to(device), target.to(device)
            output = model(data)

            data_list.append(data.cpu().numpy())
            target_list.append(target.cpu().numpy())
            output_list.append(output.cpu().numpy())
            
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    
    np.savez(SAVE_DIR / f"test_results-{TIMESTAMP}.npz", inputs=np.array(data_list), target=np.array(target_list), outputs=np.array(output_list))

    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

use_accel = not NO_ACCEL and torch.cuda.is_available()

torch.manual_seed(SEED)

if use_accel:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_kwargs = {'batch_size': TEST_BATCH_SIZE}
if use_accel:
    accel_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    test_kwargs.update(accel_kwargs)

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

val_dataset = dl._data_val

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


model = torch.load(f"{MODELS_DIR}/savar_mlp-{TIMESTAMP}.pt", map_location=device)

test(model, device, val_loader)

