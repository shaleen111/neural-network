from activations import *
from abc import ABC, abstractmethod

class Cost(ABC):
    @staticmethod
    @abstractmethod
    def delta(result, expected, z):
        pass

class MeanSquaredError(Cost):
    @staticmethod
    def delta(result, expected, z):
        return (result - expected) * Sigmoid.prime(z)


class CrossEntropyLoss(Cost):
    @staticmethod
    def delta(result, expected, z):
        return (result - expected)
