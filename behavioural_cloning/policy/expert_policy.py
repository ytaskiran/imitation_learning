from torch.nn.modules import linear
from behavioural_cloning.utils.auxiliary import *
from behavioural_cloning.policy.base_policy import BasePolicy

import numpy as np
import torch
from torch import nn 
import pickle


def build_linear_layer(W : np.float32, b : np.float32) -> nn.Linear:
    output, input = W.shape
    linear_layer = nn.Linear(input, output)
    linear_layer.weight.data = convert_to_tensor(W.T)
    linear_layer.bias.data = convert_to_tensor(b[0])

    return linear_layer


def read_layer(layer):
    assert(list(layer.keys()) == ["AffineLayer"])
    assert(sorted(layer["AffineLayer"].keys()) == ["W", "b"])

    return layer["AffineLayer"]["W"].astype(np.float32), layer["AffineLayer"]["b"].astype(np.float32)


class ExpertPolicy(BasePolicy, nn.Module):
    def __init__(self, filename, **kwargs):
        super().__init__()

        with open(filename, "rb") as f:
            data = pickle.loads(f.read())
        
        activ_type = data["nonlin_type"]

        if activ_type == "lrelu":
            self.activ = nn.LeakyReLU(0.01)
        elif activ_type == "tanh":
            self.activ = nn.Tanh()
        else:
            raise NotImplementedError()

        policy_type = [k for k in data.keys() if k != "nonlin_type"][0]

        assert(policy_type == "GaussianPolicy", f"Policy type {policy_type} not supported")

        self.policy_params = data[policy_type]

        obsnorm_mean = self.policy_params["obs_norm"]["Standardizer"]["mean_1_D"]
        obsnorm_meansq = self.policy_params["obs_norm"]["Standardizer"]["meansq_1_D"]
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))

        self.obs_norm_mean = nn.Parameter(convert_to_tensor(obsnorm_mean))
        self.obs_norm_std = nn.Parameter(convert_to_tensor(obsnorm_stdev))
        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        assert (list(self.policy_params['hidden'].keys()) == ['FeedforwardNet'])
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            layer = layer_params[layer_name]
            W, b = read_layer(layer)
            linear_layer = build_linear_layer(W, b)
            self.hidden_layers.append(linear_layer)

        # Output layer
        W, b = read_layer(self.policy_params['out'])
        self.output_layer = build_linear_layer(W, b)
        
    
    def forward(self, obs):
        normed_obs = (obs - self.obs_norm_mean) / (self.obs_norm_std + 1e-6)
        h = normed_obs
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.activ(h)
        return self.output_layer(h)

    
    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        raise NotImplementedError("""
            This policy class simply loads in a particular type of policy and
            queries it. Do not try to train it.
        """)

    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        observation = convert_to_tensor(observation.astype(np.float32))
        action = self(observation)
        return convert_to_numpy(action)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

