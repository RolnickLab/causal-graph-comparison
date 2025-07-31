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
from accelerate import Accelerator
from causal_graph_comparison import PLATFORM

# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
from causal_graph_comparison import *
from datetime import datetime
from causal_graph_comparison.utils import flatten_data_target

if PLATFORM == "cluster":
    cpu = False    
else:
    cpu = True
accelerator = Accelerator(log_with="wandb", cpu=cpu)


def train(params: dict, model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int):
    """
    Training loop for 
    data shape: torch.Size([batch_size, tau, 1, dimensions])
    target shape: torch.Size([batch_size, 1, future_timesteps, dimensions])
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

        data, target = data.to(device), target.to(device)
        # Debug shapes
        if batch_idx == 0 and epoch == 1:
            print(f"Training {model.name}...")
            print(f"Batch idx: {batch_idx}, Epoch: {epoch}")
            print(f"Data shape: {data.shape}")
            print(f"Target shape: {target.shape}")
        
        optimizer.zero_grad()
        output = model(data)

        if model.name == "lstm":
            output = output[0]

        loss = F.mse_loss(output, target)

        # Debug loss
        if batch_idx == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        loss.backward()
        optimizer.step()

        # log loss
        wandb.log({"loss_train": loss.item() / len(data)})

        if batch_idx % params["common"]["training_params"]["log_interval"] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model: nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader):
    """
    Testing loop for
    args:
        model: model to test
        device: device to test on
        test_loader: dataloader for testing data
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if model.name == "lstm":
                output = output[0]

            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}\n")
    wandb.log({"loss_valid": test_loss})

    return test_loss


def run_trainer(datamodule, model, params, dataset_type, modes, difficulty, seed, device):
    train_loader = iter(datamodule.train_dataloader(accelerator=accelerator))
    test_loader = iter(datamodule.val_dataloader())

    optimizer = optim.Adam(model.parameters(), lr=params["common"]["training_params"]["learning_rate"])

    scheduler = StepLR(optimizer, step_size=1, gamma=params["common"]["training_params"]["gamma"])

    # basline test with untrained model
    best_test_loss = test(model, test_loader)

    save_name = f"{model.name}-{dataset_type}-modes_{modes}-diff_{difficulty}-seed_{seed}"

    epochs = params["common"]["training_params"]["num_epochs"]
    for epoch in range(1, epochs + 1):
        train(params, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model, f"{MODELS_DIR}/{save_name}.pt")
        torch.save(model, f"{MODELS_DIR}/final-{save_name}.pt")
