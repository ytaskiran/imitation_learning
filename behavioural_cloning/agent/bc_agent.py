from .base_agent import BaseAgent
from behavioural_cloning.policy.policy_network import Policy
from behavioural_cloning.utils.replay_buffer import ExperienceReplayBuffer


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
        self.max_buffer_size = agent_params["max_replay_buffer_size"]

        self.policy = Policy(self.ac_dim,
                             self.ob_dim,
                             self.n_layers,
                             self.size,
                             discrete=self.discrete,
                             lr=self.learning_rate)

        self.replay_buffer = ExperienceReplayBuffer(self.max_buffer_size)

    
    def train(self, observations, actions):
        log = self.policy.update(observations, actions)
        return log

    
    def add_to_buffer(self, rollouts):
        self.replay_buffer.add_rollouts(rollouts)

    
    def sample(self, batch_size):
        return self.replay_buffer.sample_data(batch_size)


    def save(self, file_path):
        self.policy.save(file_path)