import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, 
        name, checkpoint_dir='tmp/ddpg', device='cpu'):
        super (NeuralNetwork, self).__init__()

        self.learn_rate = learn_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Hidden Layer 1
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) 
        self.bn1 = nn.LayerNorm(self.fc1_dims) 

        # Hidden Layer 2
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.bn2 = nn.LayerNorm(self.fc2_dims) 

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.checkpoint_file = os.path.join(checkpoint_dir, name+'_ddpg')
        self.device = device
        self.to(self.device)

    
    def _init_hl_weights(self):
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(NeuralNetwork):
    def __init__(self, learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, 
        name, checkpoint_dir='tmp/ddpg', device='cpu'):
        super (CriticNetwork, self).__init__(learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, 
        name, checkpoint_dir, device)

        # Action value Layer
        self.fc_act = nn.Linear(self.n_actions, fc2_dims)
        # Oputput Layer
        self.q = nn.Linear(self.fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.q.bias.data, -0.003, 0.003)
    
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = nn.functional.relu(state_value)

        state_value = self.fc2(state_value)
        action_value = self.fc_act(action)

        state_action_value = nn.functional.relu(self.bn2(torch.add(state_value, action_value)))
        state_action_value = self.q(state_action_value)

        return state_action_value

class ActorNetwork(NeuralNetwork):
    def __init__(self, learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, 
        name, checkpoint_dir='tmp/ddpg', device='cpu'):
        super (ActorNetwork, self).__init__(learn_rate, input_dims, fc1_dims, fc2_dims, n_actions, 
        name, checkpoint_dir, device)

        self.mu = nn.Linear(self.fc2_dims, n_actions)
        nn.init.uniform_(self.mu.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.mu.bias.data, -0.003, 0.003)


    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = torch.tanh(self.mu(x))

        # TODO:multiply by the bounds
        return x

