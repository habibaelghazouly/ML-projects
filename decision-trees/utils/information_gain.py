import numpy as np
from .entropy import entropy


def information_gain(y, y_left, y_right):
    E = entropy(y)  # Entropy of parent
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        return 0

    E_children = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(
        y_right
    )  # Entropy of children

    return E - E_children  # Information Gain
