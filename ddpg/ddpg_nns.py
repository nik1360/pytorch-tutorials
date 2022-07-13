'''
This files contains the class related to the Neural Networks used in the DDPG architecture.
DDPG uses an Actor network and a Critic network, structured as follows

actor:   state  -> FC1 -> LN1 -> ReLU -> FC2 -> LN2 -> ReLU -> FC3 ('mu' in the code) -> tanh -> action

critic:  state  -> FC1 -> LN1 -> ReLU -> FC2 -> LN2 -> + -> ReLU -> FC3 ('q' in the code) -> state_action_value
                                                     ^
                                                     |
                                                    ReLU
                                                     ^
                                                     |
                                                   FC_act
                                                     ^
                                                     |
                                                   action      
The networks parameter are associated as described in the DDPG paper "Continuous control with 
deep reinforcement learning".
'''
import torch
import torch.nn as nn
from torch.nn.functional import relu, tanh
import numpy as np

'''
At least in the first part, both Actor nad Critic shares the same strucure, this class is handy to define the two networks 
without rewriting same code.
'''
class NeuralNetwork(nn.Module):
    def __init__(self, learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, normalize, name):
        super(NeuralNetwork, self).__init__()
        self.learn_rate = learn_rate # Learning rate
        self.input_dims = input_dims # Number of inputs in the network
        self.fc1_dims = fc1_dims # Number of neurons in the Fully Connected layer 1
        self.fc2_dims = fc2_dims # Number of neurons in the Fully Connected layer 2
        self.n_actions = n_actions # dimension of the action space that the Agent can take
        self.name = name # Name of the Network (useful to save the model)
        self.normalize = normalize # Apply Layer Normalization

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # Fully Connected layer 1
        self.ln1 = nn.LayerNorm(self.fc1_dims) # Layer 1 Normalization

        self.fc2 = nn.Linear(*self.input_dims, self.fc2_dims) # Fully Connected layer 2
        self.ln2 = nn.LayerNorm(self.fc2_dims) # Layer 2 Normalization

        self._init_layer(layer=self.fc1, bound = 1./np.sqrt(self.fc1.weight.data.size()[0]))
        self._init_layer(layer=self.fc2, bound = 1./np.sqrt(self.fc2.weight.data.size()[0]))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    '''
    Function which initialize the layer weights of the NN according to DDPG paper 
    '''
    def _init_layer(self, layer, bound):
        nn.init.uniform_(layer.weight.data, -bound, bound)
        nn.init.uniform_(layer.bias.data, -bound, bound)


class CriticNetwork(nn.Module):
    def __init__(self, learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, normalize, name):
        super(CriticNetwork, self).__init__(learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, normalize, name)

        self.fc_act = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        self._init_layer(layer=self.q, bound = 0.003)


