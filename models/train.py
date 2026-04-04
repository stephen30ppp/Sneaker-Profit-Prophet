"""
Model Training Pipeline

Handles train/val/test splitting, training loop with early stopping,
and model checkpoint saving.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, MODEL_DIR


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "lstm",
) -> dict:
    """
    Train a model with early stopping.

    Args:
        model: PyTorch model (SneakerLSTM or SneakerGRU)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        model_name: Name for checkpoint file

    Returns:
        Dictionary with training history {train_loss: [], val_loss: []}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history
