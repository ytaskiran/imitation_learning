from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for agent.

    Inherits ABC abstract metaclass and provides an API for child classes.
    Template class for the main behavioural cloning agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def train(self): # TODO Add docstring and explanations to all functions 
        raise NotImplementedError

    @abstractmethod
    def add_to_replay_buffer(self, paths): # TODO Indicate variable type with :
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError
