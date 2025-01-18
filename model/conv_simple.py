import torch
import torch.nn as nn
import numpy as np
from layers.normalization import LayerNormalization

from config import Config

class ConvSimple(nn.Module):
    def __init__(self, in_maps, hidden_maps, out_maps, board_size, dropout_rate = 0.1):
        super(ConvSimple,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_maps, hidden_maps, 5, padding='same'),
            LayerNormalization(axis=[1]),
            nn.LeakyReLU(0.2),
            #nn.Dropout(dropout_rate),
            # nn.Conv2d(hidden_maps, hidden_maps, 5, padding='same'),
            # #LayerNormalization(axis=[1]),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_maps, hidden_maps, 3, padding='same'),
            LayerNormalization(axis=[1]),
            nn.LeakyReLU(0.2),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.logits_head = nn.Conv2d(hidden_maps, out_maps, 1, padding='same')

        self.hidden_maps = hidden_maps
        self.board_size = board_size

    def forward(self, inputs):
        x = self.main(inputs)
        x = self.dropout(x)
        logits = self.logits_head(x)
        return logits

