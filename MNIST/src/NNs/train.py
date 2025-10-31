import numpy as np
import torch
from .helpers import save_checkpoint, load_checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=10,
    device="cpu",
    early_stopping_patience=3,
    min_delta=0.001,
    checkpoint_path=None,
):
    train_loss_mean, train_loss_std = [], []
    val_loss_mean, val_loss_std = [], []
    train_acc_mean, train_acc_std = [], []
    val_acc_mean, val_acc_std = [], []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        batch_train_losses, batch_train_accs = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            batch_train_accs.append((predicted == labels).float().mean().item())

        # Epoch metrics
        train_loss_mean.append(np.mean(batch_train_losses))
        train_loss_std.append(np.std(batch_train_losses))
        train_acc_mean.append(np.mean(batch_train_accs))
        train_acc_std.append(np.std(batch_train_accs))

        # Validation
        model.eval()
        batch_val_losses, batch_val_accs = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                batch_val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                batch_val_accs.append((predicted == labels).float().mean().item())

        val_loss_mean.append(np.mean(batch_val_losses))
        val_loss_std.append(np.std(batch_val_losses))
        val_acc_mean.append(np.mean(batch_val_accs))
        val_acc_std.append(np.std(batch_val_accs))

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss_mean[-1]:.4f} | "
            f"Train Acc: {train_acc_mean[-1]*100:.2f}% | "
            f"Val Loss: {val_loss_mean[-1]:.4f} | "
            f"Val Acc: {val_acc_mean[-1]*100:.2f}%"
        )

        history = {
            "train_loss_mean": train_loss_mean,
            "train_loss_std": train_loss_std,
            "train_acc_mean": train_acc_mean,
            "train_acc_std": train_acc_std,
            "val_loss_mean": val_loss_mean,
            "val_loss_std": val_loss_std,
            "val_acc_mean": val_acc_mean,
            "val_acc_std": val_acc_std,
        }

        # Early Stopping
        current_val_loss = val_loss_mean[-1]
        if best_val_loss - current_val_loss > min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0  # reset patience
            print(f"New best model found (val_loss={best_val_loss:.4f})")

            # Save best model checkpoint
            if checkpoint_path:
                save_checkpoint(model, optimizer, epochs, history, checkpoint_path)
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break
    return history
