import matplotlib.pyplot as plt
import pandas as pd


def plot_train_val_accuracy(results_df, min_samples_split_fixed):
    """
    Plots training and validation accuracy vs max_depth for a fixed min_samples_split.
    Also prints a nicely formatted summary table.

    results_df: DataFrame with columns ['max_depth', 'min_samples_split', 'train_acc', 'val_acc']
    min_samples_split_fixed: the value of min_samples_split to filter the results
    """
    # Filter results and sort
    subset = results_df[
        results_df["min_samples_split"] == min_samples_split_fixed
    ].sort_values("max_depth")

    # Print summary table
    print(f"\nAccuracy Summary (min_samples_split={min_samples_split_fixed})")
    print(f"{'Max Depth':>10} | {'Train Accuracy':>15} | {'Validation Accuracy':>20}")
    print("-" * 50)
    for _, row in subset.iterrows():
        print(
            f"{int(row['max_depth']):>10} | {row['train_acc']*100:>14.2f}% | {row['val_acc']*100:>19.2f}%"
        )

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        subset["max_depth"], subset["train_acc"], marker="o", label="Train Accuracy"
    )
    plt.plot(
        subset["max_depth"], subset["val_acc"], marker="o", label="Validation Accuracy"
    )
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title(
        f"Train vs Validation Accuracy (min_samples_split={min_samples_split_fixed})"
    )
    plt.xticks(subset["max_depth"])
    # Dynamic Y-axis based on min/max values with a margin
    y_min = max(0, subset[["train_acc", "val_acc"]].min().min() - 0.02)
    y_max = min(1.2, subset[["train_acc", "val_acc"]].max().max() + 0.02)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_tree_complexity(results_df, min_samples_split_fixed):
    subset = results_df[
        results_df["min_samples_split"] == min_samples_split_fixed
    ].sort_values("max_depth")

    plt.figure(figsize=(8, 5))
    plt.plot(
        subset["max_depth"], subset["train_acc"], marker="o", label="Train Accuracy"
    )
    plt.plot(
        subset["max_depth"], subset["val_acc"], marker="o", label="Validation Accuracy"
    )
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title(f"Tree Complexity Analysis (min_samples_split={min_samples_split_fixed})")
    plt.xticks(subset["max_depth"])
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nSummary Table:")
    print(subset[["max_depth", "train_acc", "val_acc"]].to_string(index=False))


def plot_overfitting(results_df, min_samples_split_fixed):
    subset = results_df[
        results_df["min_samples_split"] == min_samples_split_fixed
    ].sort_values("max_depth")

    plt.figure(figsize=(8, 5))
    plt.plot(
        subset["max_depth"],
        subset["train_acc"] - subset["val_acc"],
        marker="o",
        color="red",
    )
    plt.xlabel("Max Depth")
    plt.ylabel("Train - Validation Accuracy")
    plt.title(f"Overfitting Analysis (min_samples_split={min_samples_split_fixed})")
    plt.xticks(subset["max_depth"])
    plt.grid(True)
    plt.show()

    print("\nOverfitting Gap (Train - Validation Accuracy):")
    print(
        subset[["max_depth", "train_acc", "val_acc"]]
        .assign(gap=subset["train_acc"] - subset["val_acc"])
        .to_string(index=False)
    )
