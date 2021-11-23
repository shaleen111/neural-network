from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

vector = npt.NDArray[np.float64]

class Activation(ABC):
    @staticmethod
    @abstractmethod
    def fn(z: vector) -> vector:
        pass

    @staticmethod
    @abstractmethod
    def prime(z: vector) -> vector:
        pass

class Sigmoid(Activation):
    @staticmethod
    def fn(z: vector) -> vector:
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def prime(z: vector) -> vector:
        return Sigmoid.fn(z)*(1-Sigmoid.fn(z))
