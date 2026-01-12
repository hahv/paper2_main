from abc import ABC, abstractmethod

class BaseRule(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def check(self, data_dict) -> bool:
        pass # Returns True if the rule condition is met.