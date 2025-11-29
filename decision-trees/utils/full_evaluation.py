from utils.evaluate_tree import evaluate_tree
from utils.feature_importance import compute_feature_importance
from utils.plotting import plot_tree_complexity, plot_overfitting


def run_full_evaluation(
    models,
    X_test,
    y_test,
    feature_names,
    class_names,
    results_df
):
    """
    Full evaluation for ALL trained models from hyperparameter tuning.
    """

    print("\n================ FULL EVALUATION FOR ALL MODELS ================\n")

    for entry in models:
        tree = entry["tree"]
        max_depth = entry["max_depth"]
        min_split = entry["min_samples_split"]

        print("\n-------------------------------------------------------------")
        print(f" MODEL: max_depth={max_depth}, min_samples_split={min_split}")
        print("-------------------------------------------------------------\n")

        evaluate_tree(tree, X_test, y_test, class_names=class_names)

        print("\n\n================ FEATURE IMPORTANCE ================\n")
        compute_feature_importance(tree, feature_names=feature_names)
        print("\n\n================ TREE COMPLEXITY ANALYSIS ================\n")
        plot_tree_complexity(results_df, min_samples_split_fixed=min_split)
        print("\n\n================ OVERFITTING ANALYSIS ================\n")
        plot_overfitting(results_df, min_samples_split_fixed=min_split)
    print("\n================ END OF FULL EVALUATION ================\n")