import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from causal_graph_comparison import PLATFORM

# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
from causal_graph_comparison import *

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
                    len(train_loader._dataset),
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

            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader._dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}\n")
    wandb.log({"loss_valid": test_loss})

    return test_loss


def run_trainer(datamodule, model, params, dataset_type, modes, difficulty, seed, device):
    save_name = f"{model.name}-modes_{modes}-diff_{difficulty}-seed_{seed}"

    # check if model already exists
    model_path = MODELS_DIR / f"{save_name}.pt"
    if model_path.exists():
        print(f"=== SKIPPING TRAINING: Model already exists at {model_path}")
        return model_path

    train_loader = iter(datamodule.train_dataloader(accelerator=accelerator))
    test_loader = iter(datamodule.val_dataloader())

    optimizer = optim.Adam(model.parameters(), lr=params[model.name]["training_params"]["learning_rate"])

    scheduler = StepLR(optimizer, step_size=1, gamma=params[model.name]["training_params"]["gamma"])

    # basline test with untrained model
    best_test_loss = test(model, device, test_loader)    

    epochs = params[model.name]["training_params"]["num_epochs"]
    for epoch in range(1, epochs + 1):
        train(params, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model, model_path)
        torch.save(model, f"{MODELS_DIR}/final-{save_name}.pt")

    return model_path
