import nn.neural_network as nn
import nn.activations as a
import nn.cost_functions as c
import numpy as np
import pickle


def main():
    # create neural network
    digit_recognizer = nn.NeuralNetwork([784, 50, 10], a.Sigmoid,
                                        c.CrossEntropyLoss)

    # load mnist data
    with open("mnist_training.pkl", "rb") as d:
        all_data = pickle.load(d)
    training, test = all_data[:50000], all_data[50000:]

    # vectorize the training and test sets
    # for training need to also vectorize output
    # but not for test set
    training = [(np.reshape(inp, (784, 1))/255, v_out(out))
                for inp, out in training]

    test = [(np.reshape(inp, (784, 1))/255, out) for inp, out in test]
    # digit_recognizer.load_network("network.pkl")
    # correct = digit_recognizer.perf_check(test)
    # print(f"{correct}/10000")
    digit_recognizer.SGD(training, 30, 10, 0.50, test=test)
    digit_recognizer.save_network("MSE.pkl")


def v_out(out):
    arr_out = np.zeros((10, 1))
    arr_out[out] = 1.0
    return arr_out


if __name__ == "__main__":
    main()
    # n = nn.NeuralNetwork([784, 30, 10])
    # training, validation, test = m.load_data_wrapper()
    # n.SGD(training, 30, 10, 3.0, test=test)
