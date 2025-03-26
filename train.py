from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from network import AsterPredNet

train_config = {
    "lr": 0.0001,
    "batch_size": 50,
    "num_epochs": 1000,
    "criterion": nn.MSELoss,
    "optimizer": optim.SGD,
    "scheduler": optim.lr_scheduler.StepLR,
    "save_dir": "./Models/",
    "log_dir": "./logs/",
    "resume_from": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def train_model(
    train_loader: DataLoader,
    model: AsterPredNet,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    log_dir: str,
    resume_from: str | None
) -> None:
    """
    Train the model on the given dataset.
    Args:
        train_loader: DataLoader for the training set.
        model: AsterPredNet model to be trained.
        criterion: Loss function to be used.
        optimizer: Optimizer to be used.
        scheduler: Learning rate scheduler to be used.
        num_epochs: Number of epochs to train the model.
        device: Device to be used for training.
        save_dir: Directory to save the trained model.
        log_dir: Directory to save the training logs.
        resume_from: Path to the saved model to resume training from.
    """
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from))

    train_loss_list = []
    print(f"Training model on {device}")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        loss_log = running_loss/len(train_loader)
        train_loss_list.append(loss_log)
        print(f"Epoch {epoch+1}, Loss: {loss_log}")

    print(f"Training completed in {time.time()-start_time} seconds")

    torch.save(model.state_dict(), f"{save_dir}n2_model.pth")

    plt.plot(
        range(num_epochs),
        train_loss_list,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.savefig(f"{log_dir}n2_train_loss.png")


def evaluate_model(
    test_loader: DataLoader,
    model: AsterPredNet,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the given dataset.
    Args:
        test_loader: DataLoader for the test set.
        model: AsterPredNet model to be evaluated.
        criterion: Loss function to be used.
        device: Device to be used for evaluation.
    Returns:
        The average loss over the test set.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            running_loss += loss.item()

    loss_log = running_loss/len(test_loader)
    print(f"Test Loss: {loss_log}")

    return running_loss/len(test_loader)

if __name__ == "__main__":
    from datasets import StateDataset
    # Load the data
    train_dataset = StateDataset("./Datas/Rk4_train.h5")
    test_dataset = StateDataset("./Datas/Rk4_test.h5")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False
    )
    model = AsterPredNet(
        d_model=64,
        x_len=train_dataset.x_len,
        seq_len=train_dataset.n_body,
        num_layers=2,
    )
    model.to(train_config["device"])
    criterion = train_config["criterion"]()
    optimizer = train_config["optimizer"](
        model.parameters(),
        lr=train_config["lr"],
        momentum=0.9,
        weight_decay=0.0001,
    )
    scheduler = train_config["scheduler"](optimizer, step_size=200, gamma=0.5)

    # Print she some information
    print("Training set size: ", len(train_dataset))
    print("Test set size: ", len(test_dataset))
    print("epochs: ", train_config["num_epochs"])

    train_model(
        train_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        train_config["num_epochs"],
        train_config["device"],
        train_config["save_dir"],
        train_config["log_dir"],
        train_config["resume_from"]
    )

    # Epoch 1000, Loss: 0.1494257728755474

    evaluate_model(
        test_loader,
        model,
        criterion,
        train_config["device"],
    )
    #Test Loss: 0.07299363522213839

    #Test Loss: 81.16466903686523
