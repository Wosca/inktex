"""Independent scratch initialization of PyTorch's cloned transformer blocks."""
from torch import nn


def initialize_transformer(transformer):
    for layer in transformer.layers:
        for parameter in layer.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
