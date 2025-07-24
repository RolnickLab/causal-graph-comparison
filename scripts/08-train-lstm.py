import argparse
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from causal_graph_comparison.lstm import Lstm_sine, Lstm
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from causal_graph_comparison.utils import get_json_config
from causal_graph_comparison import *
from causal_graph_comparison.test_dataset import SineTestDataset
from causal_graph_comparison.utils import flatten_data_target

# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
from datetime import datetime

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
TAU = 30
INPUT_SIZE = LATITUDE * LONGITUDE * TAU
OUTPUT_SIZE = LATITUDE * LONGITUDE * FUTURE_TIMESTEPS
HIDDEN_SIZE = 1600  # LSTM
DIFFICULTY = "med_easy"
NUM_MODES = 4
BEST = False
MODEL_TYPE = "lstm"
DATASET_TYPE = "savar"

best_name = "best-" if BEST else ""

diff_mapping = {
    "easy": "e",
    "med_easy": "me",
    "med_hard": "mh",
    "hard": "h",
}


wandb.init(
    project="climatem",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "tau": TAU,
        "future_timesteps": FUTURE_TIMESTEPS,
        "model": "lstm",
        "layers": [HIDDEN_SIZE],
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "seed": SEED,
        "gamma": GAMMA,
    },
)


def train(log_interval, dry_run, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        # Debug shapes
        if batch_idx == 0 and epoch == 1:
            print(f"Data shape: {data.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")

        optimizer.zero_grad()

        # Savar data
        # convert data to (batch_size, tau, input_size)
        data = data.squeeze()

        output, _, _ = model(data)

        # Debug output shape
        if batch_idx == 0 and epoch == 1:
            print(f"Model output shape: {output.shape}")
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        # target shape: (batch_size, future_timesteps, 1) -> (batch_size, future_timesteps)
        target = target.squeeze() if target.dim() > 2 else target

        loss = F.mse_loss(output, target)

        # Debug loss
        if batch_idx == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
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
                    loss.item(),  # TODO fix print statement
                )
            )
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            # Savar data
            # convert data to (batch_size, tau, input_size)
            data = data.squeeze()

            output, _, _ = model(data)

            # Sine data
            # output, _, _ = model(data)

            # Reshape target to match model output (batch_size, future_timesteps) (remove last dim)
            target = target.squeeze() if target.dim() > 2 else target

            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}\n")
    wandb.log({"loss_valid": test_loss})

    return test_loss


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

train_dataset = dl._data_train
test_dataset = dl._data_val

print("train_dataset length: ", len(train_dataset))
print("test_dataset length: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =======  Train model =======

# model = Lstm_sine().to(device)

model = Lstm(hidden_size=HIDDEN_SIZE, num_layers=3).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

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
