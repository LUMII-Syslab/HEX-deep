import torch
import torch.nn as nn
class LayerNormalization(nn.Module):
    def __init__(self, axis=-1, epsilon=1e-6, subtract_mean=False, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bias = None
        self.subtract_mean = subtract_mean
        super(LayerNormalization, self).__init__(**kwargs)

    def forward(self, inputs):
        variance = torch.mean(torch.square(inputs), self.axis, keepdim=True)
        return inputs * torch.rsqrt(variance + self.epsilon)
