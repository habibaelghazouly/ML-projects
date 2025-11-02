import matplotlib.pyplot as plt
import numpy as np

def plot_curves(train_values, val_values, title, ylabel):
    plt.figure()
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def print_confusion_matrix(cm, classes=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
