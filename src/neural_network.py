'''
This module implements a simple neural network.
'''

import pickle
from random import shuffle
from typing import List, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from activations import Activation
from cost_functions import Cost 

class NeuralNetwork:
    '''
    A class used to represent neural networks
    
    Args:
        layers (List[int]): A list of integers describing the size of each layer of the network.
        activation (Activation): Activation function to be used by the neurons of the network.
        cost (Cost): The cost/loss function that the network will try to minimize. 
    
    Attributes:
        num_of_layers (int): Number of layers of the network.
        layers (List[int]): This is where we store layers.
        activation (Activation): This is where activation is stored.
        cost (Cost): This where cost is stored.
        bias (list[npt.NDArray[np.float64]]): A list whose i'th entry gives you an ndarray of the
                                              biases of the (i + 1)'th layer
        weights (list[npt.NDArray[np.float64]]): A list whose i'th entry gives you a matrix (represented 
                                                 as an ndarray) of the weights of (i + 1)'th layer. The
                                                 matrix is organized such that the weight of the
                                                 connection between the j'th neuron in the i'th layer &
                                                 the k'th neuron in the (i + 1)'th layer is the (k, j)
                                                 entry of the matrix.
    '''
    def __init__(self, layers: List[int], activation: Activation, cost: Cost):
        self.num_of_layers: int = len(layers)
        self.layers: List[int] = layers

        self.activation: Activation = activation
        self.cost: Cost = cost

        self.bias: list[npt.NDArray[np.float64]] = [np.random.randn(layer, 1) 
                                                    for layer in layers[1:]]

        self.weights: list[npt.NDArray[np.float64]] = [np.random.randn(layers[i], layers[i-1]) 
                                                       for i in range(1, self.num_of_layers)]

    def load_network(self, path):
        with open(path, "rb") as pkl:
            self.bias, self.weights, self.activation, self.cost = \
                pickle.load(pkl)

    def save_network(self, path):
        network = (self.bias, self.weights, self.activation, self.cost)
        with open(path, "wb") as pkl:
            pickle.dump(network, pkl)

    # input is an array indicating values of neurons in the first layer
    def forward_prop(self, inp):
        for b, w in zip(self.bias, self.weights):
            inp = self.activation.fn(np.dot(w, inp) + b)
        return inp

    # returns z and activations for all neurons given
    # some input
    def get_network_state(self, inp):
        layers = [inp]
        interim_val = []
        for (b, w) in zip(self.bias, self.weights):
            interim_val.append(np.dot(w, inp) + b)
            layers.append(self.activation.fn(interim_val[-1]))
            inp = layers[-1]
        return (interim_val, layers)

    def back_prop(self, inp, out):
        db = [np.zeros(b.shape) for b in self.bias]
        dw = [np.zeros(w.shape) for w in self.weights]

        prev_layer = inp
        layers = [prev_layer]
        interim_val = []
        for b, w in zip(self.bias, self.weights):
            # print(w.shape, b.shape)
            z = np.dot(w, prev_layer) + b
            interim_val.append(z)
            prev_layer = self.activation.fn(z)
            layers.append(prev_layer)

        # interim_val, layers = self.get_network_status(inp)

        # Calculate error in the last layer
        delta = self.cost.delta(layers[-1], out, interim_val[-1])
        db[-1] = delta
        dw[-1] = np.dot(delta, layers[-2].transpose())

        # backprop starting from 2nd to last layer
        # to 2nd layer
        for i in range(2, self.num_of_layers):
            z = interim_val[-i]
            delta = np.dot(self.weights[-i + 1].transpose(),
                           delta) * self.activation.prime(z)
            db[-i] = delta
            dw[-i] = np.dot(delta, layers[-i - 1].transpose())

        return (db, dw)

    # l for lambda
    def train(self, training, epochs, batch_size, learning_rate, l, test=None):
        for epoch in range(epochs):
            # In each epoch divide the data into randomized groups
            # of size batch_size
            shuffle(training)
            batches = [training[i:i + batch_size] for i in
                       range(0, len(training), batch_size)]

            # Update weights and biases by running gradient
            # descent on each mini batch
            for batch in batches:
                self.update_batch(batch, learning_rate, l, len(training))

            if test:
                print(f"{self.perf_check(test, False)}/{len(test)} succeeded")
            print(f"Epoch {epoch} completed successfully.")

    def update_batch(self, batch, learning_rate, l, n):
        # two arrays to keep track of the sum of
        # del b and del w that is computed for each
        # input so we can avg it later

        sum_db = [np.zeros(b.shape) for b in self.bias]
        sum_dw = [np.zeros(w.shape) for w in self.weights]

        for inp, out in batch:
            db, dw = self.back_prop(inp, out)
            sum_db = [s+nb for s, nb in zip(sum_db, db)]
            sum_dw = [s+nb for s, nb in zip(sum_dw, dw)]

        decay_factor = 1 - learning_rate * l / n
        # print(sum_db)
        # update weights & biases based on changes suggested by backprop
        self.weights = [decay_factor * w - (learning_rate/len(batch)) * s
                        for s, w in zip(sum_dw, self.weights)]
        self.bias = [b - (learning_rate/len(batch)) * s
                     for s, b in zip(sum_db, self.bias)]

        # Create matrices where each column is a vector in the minibatch
        # Do this by first reducing the vectors into 1D arrays
        # and then transposing the matrix
        # inp = []
        # out = []

        # for i, o in batch:
        #     inp.append(i)
        #     out.append(o)

        # inp = np.asarray(inp).transpose()
        # out = np.asarray(out).transpose()

    def perf_check(self, test_set, dump):
        result = [(np.argmax(self.forward_prop(inp)), out)
                  for (inp, out) in test_set]

        if dump:
            for a in range(len(result)):
                if result[a][0] != result[a][1]:
                    img = np.reshape(test_set[a][0], (28, 28)) * 255
                    img = Image.fromarray(img.astype('uint8'), 'L')
                    img.save(f"{result[a][0]}nf{result[a][1]}.png")

        return sum(int(nn == actual) for (nn, actual) in result)
