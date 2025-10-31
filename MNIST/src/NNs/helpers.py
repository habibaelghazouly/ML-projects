import torch
import os


def save_checkpoint(model, optimizer, epoch, history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        path,
    )
    print(f"Checkpoint saved at: {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    model.to(device)
    print(f"Checkpoint loaded from {path}, resuming at epoch {start_epoch}")
    return model, optimizer, start_epoch, history
