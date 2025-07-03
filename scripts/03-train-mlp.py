from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from climatem.data_loader.causal_datamodule import CausalClimateDataModule

# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from causal_graph_comparison import MODELS_DIR, SCRATCH_DIR
from causal_graph_comparison.mlp import Net
from causal_graph_comparison.utils import flatten_data_target

# Training settings
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 1.0
GAMMA = 0.7
NO_ACCEL = False
DRY_RUN = False
SEED = 1
LOG_INTERVAL = 10
SAVE_MODEL = True
FUTURE_TIMESTEPS = 5
LONGITUDE = 40
LATITUDE = 40
TAU = 5
LAYERS = [1600, 800, 1600]
INPUT_SIZE = LATITUDE * LONGITUDE * TAU
OUTPUT_SIZE = LATITUDE * LONGITUDE * FUTURE_TIMESTEPS

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


def train(log_interval, dry_run, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # flatten the data and target
        data, target = flatten_data_target(data, target, INPUT_SIZE, OUTPUT_SIZE)

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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:

            # flatten the data and target
            data, target = flatten_data_target(data, target, INPUT_SIZE, OUTPUT_SIZE)

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}\n")
    wandb.log({"loss_valid": test_loss})


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
val_dataset = dl._data_val

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Net(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, layers=LAYERS).to(device)

optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

# basline test with untrained model
test(model, device, val_loader)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

for epoch in range(1, EPOCHS + 1):
    train(LOG_INTERVAL, DRY_RUN, model, device, train_loader, optimizer, epoch)
    test(model, device, val_loader)
    scheduler.step()

    if SAVE_MODEL:
        # TODO: save best model & config
        torch.save(model, f"{MODELS_DIR}/savar_mlp-{timestamp}.pt")
