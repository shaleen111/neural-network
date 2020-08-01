class MeanSquaredError():

    @staticmethod
    def delta(result, expected, z, activation):
        return (result - expected) * activation.prime(z)


class CrossEntropyLoss():
    pass
