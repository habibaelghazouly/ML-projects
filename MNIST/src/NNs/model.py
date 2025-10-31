import torch
import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], output_size=10):
        super().__init__()
        self.hidden_layers = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.out = nn.Linear(prev_size, output_size)

    # Sets weights ~ U(-a, a), where a = sqrt(6 / (n_in + n_out)).
    # This keeps forward and backward signal variances balanced across layers.
    # Biases are initialized to zero.
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x
