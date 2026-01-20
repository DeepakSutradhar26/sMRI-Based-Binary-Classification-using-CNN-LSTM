import torch
import torch.nn as nn
from torch.optim import NAdam
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from models.CNN_LSTM import CNN_LSTM
from architecture.cnn1 import CNNArchitecture as CNN1
from architecture.cnn2 import CNNArchitecture as CNN2
from architecture.cnn3 import CNNArchitecture as CNN3
from data_pipeline.data_loader import train_loader, val_loader

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0

    for x,y in tqdm(loader, desc="Training", leave=False):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        optimizer.zero_grad()

        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()

        epoch_loss += loss.item()

        optimizer.step()

    return epoch_loss/len(loader) if len(loader) > 0 else 0

def validate(model, loader, criterion):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(config.DEVICE) #[8, 4, 1, 128, 128, 32]
            y = y.to(config.DEVICE) #[8, 1]

            preds_over_time = []

            preds = model(x)
            preds_over_time.append(preds)

            loss = criterion(preds, y)
            epoch_loss += loss.item()

            probs = torch.sigmoid(preds)
            preds = (probs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct/total if total > 0 else 0
    return epoch_loss/len(loader) if len(loader) > 0 else 0, acc

def plot_loss_curve(train_loss, val_loss, name):
    plt.figure(figsize=(8,5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Decreasing Over Time")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{name}_loss_curve.png", dpi=300)
    plt.close()

def plot_all_loss(all_train_losses, all_val_losses):
    model_names = ["CNN1", "CNN2", "CNN3"]
    colors = ["blue", "green", "orange"]

    plt.figure(figsize=(10,6))

    for train_losses, val_losses, name, color in zip(all_train_losses, all_val_losses, model_names, colors):
        plt.plot(train_losses, color = color, linewidth=2, label=f"{name} Train")
        plt.plot(val_losses, color=color, linestyle="--",linewidth=2, label=f"{name} Val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison (All Models)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("all_models_loss_curve.png", dpi=300)
    plt.close()

def plot_accuracy_curve(acc, name):
    plt.figure(figsize=(10,6))

    plt.plot(acc, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{name}_accuracy.png", dpi=300)
    plt.close()

def plot_all_accuracy(all_acc):
    model_names = ["CNN1", "CNN2", "CNN3"]
    colors = ["blue", "green", "orange"]

    for acc, name, color in zip(all_acc, model_names, colors):
        plt.plot(acc, color=color, linewidth=2, label=f"{name}")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("All models accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig("all_models_accuracy.png", dpi=300)
    plt.close()

def train_cnn_model(model, name, all_train_losses, all_val_losses, all_acc):
    optimizer = NAdam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    acc = []

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
        acc.append(val_acc)

        print(
            f"Train Loss : {train_loss:.2f} | Val Loss : {val_loss:.2f} | Accuracy : {val_acc:.2f}"
        )
    
    plot_loss_curve(train_losses, val_losses, name)
    plot_accuracy_curve(acc, name)
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

def main():
    all_train_losses = []
    all_val_losses = []
    all_acc = []

    model1 = CNN_LSTM(CNN1).to(config.DEVICE)
    train_cnn_model(model1, "CNN1", all_train_losses, all_val_losses, all_acc)
    model2 = CNN_LSTM(CNN2).to(config.DEVICE)
    train_cnn_model(model2, "CNN2", all_train_losses, all_val_losses, all_acc)
    # model3 = CNN_LSTM(CNN3).to(config.DEVICE)
    # train_cnn_model(model3, "CNN3", all_train_losses, all_val_losses, all_acc)

    plot_all_loss(all_train_losses, all_val_losses)
    plot_all_accuracy(all_acc)

if __name__ == "__main__":
    main()