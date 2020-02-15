from numpy import random.randn, exp


class Network():
    def __init__(self, layers_size):
        self.n_layers = len(layers_size)
        self.neurons = layers_size

        # bias[layer - 1][neuron in layer]
        self.biases = [randn(n) for n in neurons[1:]]

        # w[layer - 1][neuron in layer][neuron in back layer]
        self.weights = [randn(x, y) for x, y in zip(neurons[1:] neurons)]

    # Activation Function is a Sigmoid
    def activation(z):
        return 1.0/(1.0 + exp(-z))

    def feedforward(self, a):
        pass
