import numpy as np


class Sigmoid():
    @staticmethod
    def fn(z):
        z = np.clip(z, -500, 500)
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.fn(z)*(1-Sigmoid.fn(z))
