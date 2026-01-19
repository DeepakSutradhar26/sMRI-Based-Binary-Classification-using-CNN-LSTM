import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import tqdm

import config
import models.CNN_LSTM as CNN_LSTM
from data_pipeline.data_loader import train_loader, val_loader

def train_one_epoch(model, loader, optimizer, criterian):
    model.train()
    epoch_loss = 0.0

    for x,y in tqdm(loader, desc="Training", leave=False):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterian(preds, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss/len(loader)

def validate(model, loader, criterian):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            preds = model(x)
            loss = criterian(preds, y)
            epoch_loss += loss

            preds = (preds > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct/total if total > 0 else acc = 0
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
    model = CNN_LSTM().to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterian = nn.BCELoss()

    train_loss = []
    val_loss = []

    for epoch in range(config.N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.N_EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterian
        )

        val_loss, val_acc = train_one_epoch(
            model, val_loader, criterian
        )

        train_loss.append(train_loss)
        val_loss.append(val_loss)

        print(
            f"Train Loss : {train_loss:.2f} | Val Loss : {val_loss:.2f} | Accuracy : {val_acc:.2f}"
        )

    plot_accuracy_curve(train_loss, val_loss)


if __name__ == "__main__":
    main()