from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    """Abstract base class for policy.

    Inherits ABC abstract metaclass and provides an API for child classes.
    Template class for the main policy network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_action(self, observations : np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def update(self, observations : np.ndarray, 
               actions : np.ndarray, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, file_path: str):
        raise NotImplementedError
