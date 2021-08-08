from cs285.infrastructure.pytorch_util import from_numpy
import torch
from torch import nn
from torch import optim
from torch import distributions
import numpy as np
from collections import OrderedDict
import itertools

from imitation_learning.policy.base_policy import BasePolicy
from imitation_learning.utils.auxiliary import *



class NetworkPolicy(BasePolicy, nn.Module):

    def __init__(self, ac_dim, ob_dim, n_layers, size,
                 discrete=False, lr=1e-4, **kwargs):
        super().__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.discrete = discrete
        self.learning_rate = lr

        if self.discrete:
            self.network = build_network(self.size,
                                         self.ob_dim,
                                         self.ac_dim,
                                         self.n_layers)
            self.network.to(device)
            self.optimizer = optim.Adam(self.network.parameters(),
                                        self.learning_rate)
            self.loss = nn.CrossEntropyLoss()

        else:
            self.network = build_network(self.size,
                                         self.ob_dim,
                                         self.ac_dim,
                                         self.n_layers)
            self.network.to(device)
            self.logstd = nn.Parameter(torch.zeros(self.ac_dim,
                                                   dtype=torch.float32,
                                                   device=device))
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.network.parameters()),
                self.learning_rate)
            self.loss = nn.MSELoss()

    
    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    
    def get_action(self, observation : np.ndarray) -> np.ndarray:
        if (not len(observation.shape) > 1):
            observation = observation[None] # Adds extra dimension
        
        observation = convert_to_tensor(observation.astype(np.float32))
        action = convert_to_numpy(self.forward(observation))

        return action

    
    def update(self, observations, actions, **kwargs):
        observations_tensor = convert_to_tensor(observations.astype(np.float32))
        groundtruth_actions = convert_to_tensor(actions.astype(np.float32))
        policy_actions = self.forward(observations_tensor)

        loss = self.loss(policy_actions, groundtruth_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    

    def forward(self, observation : torch.FloatTensor):
        if self.discrete:
            return self.network.forward(observation)
        else:
            mean = self.network.forward(observation)
            dist = distributions.Normal(mean, torch.exp(self.logstd))
            sample = dist.rsample()

            return sample
        

class Policy(NetworkPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(self, observations, actions):

        loss = super().update(observations, actions)
        return {
            # TODO You can add extra logging information here, but keep this line
            'Training Loss': convert_to_numpy(loss),
        }




convert_activ = {"relu" : nn.ReLU(),
                 "tanh" : nn.Tanh(),
                 "leaky_relu" : nn.LeakyReLU(),
                 "sigmoid" : nn.Sigmoid(),
                 "selu" : nn.SELU(),
                 "softplus" : nn.Softplus(),
                 "identity" : nn.Identity()}

def build_network(size, input_size, output_size, n_layers,
                  activation="tanh", activation_out="identity"):

    if isinstance(activation, str):
        activation = convert_activ[activation]
    if isinstance(activation_out, str):
        activation_out = convert_activ[activation_out]

    layers = OrderedDict()
    layers["input"] = nn.Linear(input_size, size)
    layers["inp_activ"] = activation
    for i in range(n_layers):
        layers[f"hidden{i}"] = nn.Linear(size, size)
        layers[f"hid{i}_activ"] = activation
    layers["output"] = nn.Linear(size, output_size)
    layers["out_activ"] = activation_out

    network = nn.Sequential(layers)

    return network


