import numpy as np

inputs = np.array([-1, 7, 5])
weights = np.array([0.8, 0.1, 0])

def sum(inputs, weights):
    return inputs.dot(weights)  #  dot Realiza o Produto escalar


s = sum(inputs, weights)


def stepFunction(sum):
    if (sum >= 1):
        return 1
    return 0

r = stepFunction(s)

