import numpy as np
import pickle
from random import shuffle


class NeuralNetwork:

    # layers is an array whose i'th
    # element indicates the size of the i'th
    # layer
    def __init__(self, layers):
        self.num_of_layers = len(layers)
        self.layers = layers

        # random.randn generates array of your specified dimension
        # bias is a multidimensional array whose i'th entry
        # gives you an array of the biases for the i+1'th layer
        self.bias = [np.random.randn(layer, 1) for layer in layers[1:]]

        # weights is a multidimensional array whose i'th entry
        # gives you a matrix/array of the weights for the i+1'th layer
        self.weights = [np.random.randn(layers[i], layers[i-1]) for i in
                        range(1, self.num_of_layers)]

    def load_network(self, path):
        with open(path, "rb") as pkl:
            self.bias, self.weights = pickle.load(pkl)

    def save_network(self):
        network = (self.bias, self.weights)
        with open("network.pkl", "wb") as pkl:
            pickle.dump(network, pkl)

    # input is an array indicating values of neurons in the first layer
    def forward_prop(self, inp):
        for b, w in zip(self.bias, self.weights):
            inp = self.sigmoid(np.dot(w, inp) + b)
        return inp

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
            prev_layer = self.sigmoid(z)
            layers.append(prev_layer)

        # Calculate error in the last layer
        delC_A = layers[-1] - out
        delta = delC_A * self.sigmoid_prime(interim_val[-1])
        db[-1] = delta
        dw[-1] = np.dot(delta, layers[-2].transpose())

        # backprop starting from 2nd to last layer
        # to 2nd layer
        for i in range(2, self.num_of_layers):
            z = interim_val[-i]
            delta = np.dot(self.weights[-i + 1].transpose(),
                           delta) * self.sigmoid_prime(z)
            db[-i] = delta
            dw[-i] = np.dot(delta, layers[-i - 1].transpose())

        return (db, dw)

    def SGD(self, training, epochs, batch_size, learning_rate, test=None):
        for epoch in range(epochs):
            # In each epoch divide the data into randomized groups
            # of size batch_size
            shuffle(training)
            batches = [training[i:i + batch_size] for i in
                       range(0, len(training), batch_size)]

            # Update weights and biases by running gradient
            # descent on each mini batch
            for batch in batches:
                self.update_batch(batch, learning_rate)

            if test:
                print(f"{self.perf_check(test)}/{len(test)} succeeded")
            print(f"Epoch {epoch} completed successfully.")

    def update_batch(self, batch, learning_rate):
        # two arrays to keep track of the sum of
        # del b and del w that is computed for each
        # input so we can avg it later
        sum_db = [np.zeros(b.shape) for b in self.bias]
        sum_dw = [np.zeros(w.shape) for w in self.weights]

        for inp, out in batch:
            db, dw = self.back_prop(inp, out)
            sum_db = [s+n for s, n in zip(sum_db, db)]
            sum_dw = [s+n for s, n in zip(sum_dw, dw)]

        # print(sum_db)
        # update weights & biases based on changes suggested by backprop
        self.weights = [w - (learning_rate/len(batch)) * s
                        for s, w in zip(sum_dw, self.weights)]
        self.bias = [b - (learning_rate/len(batch)) * s
                     for s, b in zip(sum_db, self.bias)]

    def perf_check(self, test_set):
        result = [(np.argmax(self.forward_prop(inp)), out)
                  for (inp, out) in test_set]
        return sum(int(nn == actual) for (nn, actual) in result)

    # sigmoid is the activation function being used
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
