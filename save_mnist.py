import mnist
import pickle


def save_mnist():
    train = mnist.train_images()
    train = train.reshape((train.shape[0], train.shape[1]*train.shape[2]))

    label = mnist.train_labels()

    data = list(zip(list(train), label))

    with open("mnist_training.pkl", "wb") as pkl:
        pickle.dump(data, pkl)


if __name__ == "__main__":
    save_mnist()
