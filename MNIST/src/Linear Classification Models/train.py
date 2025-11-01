import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def train_model(model, train_loader, val_loader, epochs, lr, device, loss_fn, binary=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device).float() if binary else y.to(device)

            optimizer.zero_grad()
            output = model(x)

            if binary:
                output = output.view(-1)
                loss = loss_fn(output, y)
                preds = (output >= 0.5).float()
            else:
                loss = loss_fn(output, y)
                preds = output.argmax(1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_losses.append(total_loss / total)
        train_accs.append(correct / total)

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device, binary)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, loader, loss_fn, device, binary=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float() if binary else y.to(device)
            output = model(x)

            if binary:
                output = output.view(-1)
                loss = loss_fn(output, y)
                preds = (output >= 0.5).float()
            else:
                loss = loss_fn(output, y)
                preds = output.argmax(1)

            total_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def test_model(model, test_loader, device, binary=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = (output.view(-1) >= 0.5).float() if binary else output.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return acc, cm
