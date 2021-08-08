from .base_agent import BaseAgent
from imitation_learning.policy.policy_network import Policy

class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        self.env = env
        self.ac_dim = agent_params["ac_dim"]
        self.ob_dim = agent_params["ob_dim"]
        self.n_layers = agent_params["n_layers"]
        self.size = agent_params["size"]
        self.discrete = agent_params["discrete"]
        self.learning_rate = agent_params["learning_rate"]

        self.policy = Policy(self.ac_dim,
                             self.ob_dim,
                             self.n_layers,
                             self.size,
                             discrete=self.discrete,
                             lr=self.learning_rate)

        