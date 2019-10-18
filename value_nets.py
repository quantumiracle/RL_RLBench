
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import argparse
import time


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, state_space, num_actions, hidden_size, device, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.backbones=Backbones(state_space, hidden_size, device)
        sum_hidden_dim=hidden_size*len(state_space)

        self.linear1 = nn.Linear(sum_hidden_dim + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state_list, action):
        state_feature = self.backbones(state_list)
        x = torch.cat([state_feature, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class VecQNetwork(nn.Module):
    def __init__(self, state_space, num_actions, hidden_size, vec_dim, device, init_w=3e-3):
        """
        vec_dim: dimension of vectorized Q value, usually according to the dimension of reward function 
        """
        super(VecQNetwork, self).__init__()
        self.backbones=Backbones(state_space, hidden_size, device)
        sum_hidden_dim=hidden_size*len(state_space)

        self.linear1 = nn.Linear(sum_hidden_dim + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, vec_dim)  # output 
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state_list, action):
        state_feature = self.backbones(state_list)
        x = torch.cat([state_feature, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    

class Backbones(nn.Module):
    """ Extract features from different types of states 

        for vector state, dim(shape)==1: MLP
        for tensor (image) state, dim(shape)>1: CNN
    """
    def __init__(self, state_space, hidden_size, device):
        super(Backbones, self).__init__()
        self.device = device
        self.hidden = nn.ModuleList()  # need to use ModuleList for various number of layers on cuda
        self.cnn_tail = nn.ModuleList()
        self.after_conv_size_list=[]

        for single_state_space in state_space:
            state_dim = single_state_space.shape[0]
            if len(single_state_space.shape) == 1:  # vector
                self.hidden.append(nn.Sequential(
                    nn.Linear(state_dim, hidden_size),
                    nn.ReLU(True)
                    ))
            elif len(single_state_space.shape) > 1: # tensor, images, etc
                assert single_state_space.shape[0]==single_state_space.shape[1] # square image
                input_channel = 3
                output_channel = 32
                after_conv_size = int(output_channel*(state_dim/4)**2) # calculate the dimension after convolution
                self.after_conv_size_list.append(after_conv_size)

                self.hidden.append( nn.Sequential(
                nn.Conv2d(input_channel, int(output_channel/2), kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.ReLU(True),
                nn.BatchNorm2d(int(output_channel/2)),

                nn.Conv2d(int(output_channel/2), output_channel, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.ReLU(True),
                nn.BatchNorm2d(output_channel),
                ))

                self.cnn_tail.append(nn.Linear(after_conv_size, hidden_size))


            else:
                raise ValueError('Wrong State Shape!')

    def forward(self, state_list):
        x=[]
        idx=0
        for (state, layer) in zip(np.rollaxis(state_list, 1), self.hidden):  # first dim is N: number of samples
            # list of array to array
            if len(state[0].shape)>1:
                state = np.vstack([state.tolist()])  # vstack lose first dimension for more than 1 dim tensor
                state = torch.FloatTensor(state).to(self.device)
                z=layer(state)
                z=z.view(-1, self.after_conv_size_list[idx])
                x.append(self.cnn_tail[idx](z))
                idx+=1
            else:
                state = np.vstack(state)
                state = torch.FloatTensor(state).to(self.device)
                x.append(layer(state))
        output = torch.cat(x, dim=-1)
        
        return output