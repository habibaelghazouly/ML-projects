import numpy as np
import matplotlib.pyplot as plt

def detect_convergence(val_loss_mean, threshold=0.005):
    vepochs = np.arange(1, len(val_loss_mean) + 1)
    loss_change = np.abs(np.diff(val_loss_mean))
    converged_epochs = np.where(loss_change < threshold)[0]
    if len(converged_epochs) > 0:
        return converged_epochs[0] + 1
    return vepochs[-1]

def plot_convergence(train_loss_mean, val_loss_mean, converged_epoch):
    vepochs = np.arange(1, len(val_loss_mean) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(vepochs, train_loss_mean, label='Train Loss', marker='o')
    plt.plot(vepochs, val_loss_mean, label='Validation Loss', marker='o')
    plt.axvline(x=converged_epoch, color='r', linestyle='--', label=f'Converged Epoch: {converged_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss with Convergence')
    plt.legend()
    plt.show()
