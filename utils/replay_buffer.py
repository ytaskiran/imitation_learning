from imitation_learning.utils.auxiliary import *
import numpy as np


class ExperienceReplayBuffer():

    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.rollouts = []

        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None


    def __len__(self):
        if self.observations:
            return self.observations.shape[0]
        else:
            return 0


    def add_rollouts(self, rollouts):
        for rollout in rollouts:
            self.rollouts.append(rollout)

        observations, actions, rewards, next_obs, terminals = concatenate_rollouts(rollouts)

        if self.observations is None:
            self.observations = observations[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_observations = next_obs[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.observations = np.concatenate([self.observations, observations])[-self.max_size:]
            self.actions = np.concatenate([self.actions, actions])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, rewards])[-self.max_size:]
            self.next_observations = np.concatenate([self.next_observations, next_obs])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]

    
    def sample_data(self, batch_size, random=True):
        assert (self.observations.shape[0]
             == self.actions.shape[0]
             == self.rewards.shape[0]
             == self.next_observations.shape[0]
             == self.terminals.shape[0])

        size = self.observations.shape[0]

        if random:
            indices = np.random.permutation(size)[:batch_size]
            return (self.observations[indices], 
                    self.actions[indices],
                    self.rewards[indices], 
                    self.next_observations[indices],
                    self.terminals[indices])
        else:
            return (self.observations[-batch_size:],
                    self.actions[-batch_size:],
                    self.rewards[-batch_size:],
                    self.next_observations[-batch_size:],
                    self.terminals[-batch_size:])

