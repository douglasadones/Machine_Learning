import numpy as np

# We're add multilayer to resolve problem with XOR

def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

outputs = np.array([
    [0],
    [1],
    [1],
    [0],
])

weights0 = np.array([[-0.424, -0.740, -0.961],
                    [0.358, -0.893, -0.469],])

weights1 = np.array([[-0.017], [-0.893], [0.148]])

epoch = 100

for j in range(epoch):
    input_layer = inputs
    synapse_sum = np.dot(input_layer, weights0)
    hidden_layer = sigmoid(synapse_sum)

    