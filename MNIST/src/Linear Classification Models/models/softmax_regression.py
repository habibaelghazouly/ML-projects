import torch
import torch.nn as nn

class SoftmaxRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # softmax will be applied in loss
