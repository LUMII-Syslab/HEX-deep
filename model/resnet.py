import torch
import torch.nn as nn
import numpy as np
from layers.normalization import LayerNormalization

from config import Config

def inv_sigmoid(y, eps=1e-10):
  y = np.maximum(y, eps)
  y = np.minimum(y, 1 - eps)
  return np.log(y / (1 - y))

class ResBlock(nn.Module):
    def __init__(self,hidden_maps):
        super(ResBlock,self).__init__()
        self.main = nn.Sequential(
            LayerNormalization(axis=[1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_maps, hidden_maps, 3, padding='same'),
            LayerNormalization(axis=[1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_maps, hidden_maps, 3, padding='same'),
        )
        self.hidden_maps = hidden_maps

    def forward(self, inputs):
        out = inputs+self.main(inputs)
        return out

class ConvMove(nn.Module):
    def __init__(self, in_maps, hidden_maps, out_maps, board_size, dropout_rate = 0.1):
        super(ConvMove,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_maps + 1 + hidden_maps, hidden_maps, 5, padding='same'),
            #LayerNormalization(axis=[2, 3]),
            #nn.LeakyReLU(0.2),
            #nn.Conv2d(hidden_maps, hidden_maps, 3, padding='same')
        )
        self.last = nn.Sequential(
            nn.Conv2d(hidden_maps, hidden_maps, 1, padding='same'),
            nn.Dropout(dropout_rate),
            LayerNormalization(axis=[1]),
            nn.LeakyReLU(0.2),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.res1 = ResBlock(hidden_maps)
        self.res2 = ResBlock(hidden_maps)
        self.res3 = ResBlock(hidden_maps)
        self.res4 = ResBlock(hidden_maps)
        self.logits_head = nn.Conv2d(hidden_maps, out_maps, 1, padding='same')
        self.state_head = nn.Conv2d(hidden_maps, hidden_maps, 1, padding='same')

        self.lr_adjust = 4
        residual_weight = np.random.uniform(size=[1, hidden_maps, 1, 1], )
        self.residual_scale_param = nn.Parameter(torch.tensor(inv_sigmoid(residual_weight) / self.lr_adjust, device=Config.device,dtype=torch.float32))
        self.initial_state_param = nn.Parameter(torch.zeros([1,hidden_maps,board_size, board_size]))

        self.hidden_maps = hidden_maps
        self.board_size = board_size
        # self.logits_head.weight.data.fill_(0.0)
        # self.logits_head.bias.data.fill_(0.0)

    def initial_state(self, batch_size):
        return torch.tile(self.initial_state_param, [batch_size,1,1,1])
        #return torch.zeros([batch_size, self.hidden_maps, self.board_size, self.board_size], device=Config.device)

    def forward(self, inputs, state):
        state_drop= self.dropout(state)
        rand = torch.randn([inputs.shape[0],1,inputs.shape[2],inputs.shape[3]], device=inputs.device)
        #rand = torch.ones([inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]], device=inputs.device)*0.1
        inputs = torch.cat([inputs, rand,state_drop], dim=1)
        x = self.conv1(inputs)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.last(x)
        logits = self.logits_head(x)*0.2#/np.sqrt(self.hidden_maps)
        cand = self.state_head(state)
        residual_scale = torch.sigmoid(self.residual_scale_param * self.lr_adjust)
        new_state = cand*residual_scale + state*(1-residual_scale)
        return logits, new_state

