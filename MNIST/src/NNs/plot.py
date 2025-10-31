import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history, epochs):
    
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(20, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss_mean"], label='Train Loss')
    plt.plot(epochs_range, history["val_loss_mean"], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["train_acc_mean"], label='Train Accuracy')
    plt.plot(epochs_range, history["val_acc_mean"], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Error Bars
    plt.subplot(1, 3, 3)
    plt.errorbar(epochs_range, history["train_loss_mean"], yerr=history["train_loss_std"], label='Train Loss', capsize=3)
    plt.errorbar(epochs_range, history["val_loss_mean"], yerr=history["val_loss_std"], label='Validation Loss', capsize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves with Error Bars')
    plt.legend()

    plt.tight_layout()
    plt.show()
