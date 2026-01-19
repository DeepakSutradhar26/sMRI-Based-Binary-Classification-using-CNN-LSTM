import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from models.CNN_LSTM import CNN_LSTM
from architecture.cnn3 import CNNArchitecture
from data_pipeline.data_loader import train_loader, val_loader

def create_one_batch(x, index):
    batch = []
    for i in range(x.shape[0]):
        batch.append(x[i][index])
    return torch.tensor(batch)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0

    for x,y in tqdm(loader, desc="Training", leave=False):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        for i in range(x.shape[1]):
            batch = create_one_batch(x, i)

            print(batch.shape)

            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss/(x.shape[0] * x.shape[1])

def validate(model, loader, criterion):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(config.DEVICE) #[8, 5, 1, 128, 128, 32]
            y = y.to(config.DEVICE) #[8, 1]

            preds = model(x)
            loss = criterion(preds, y)
            epoch_loss += loss.item()

            preds = (preds > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
    
    acc = correct/total if total > 0 else 0
    return epoch_loss/len(loader), acc

def plot_accuracy_curve(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Decreasing Over Time")
    plt.legend()
    plt.show()

def main():
    model = CNN_LSTM(CNNArchitecture).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []

    for epoch in range(config.N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.N_EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Train Loss : {train_loss:.2f} | Val Loss : {val_loss:.2f} | Accuracy : {val_acc:.2f}"
        )

    plot_accuracy_curve(train_losses, val_losses)


if __name__ == "__main__":
    main()