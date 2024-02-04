import numpy as np

# We're add multilayer to resolve problem with XOR (Non-linear problem)

def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))


def derivativeSigmoid(sig):  # This allow us to choose the right gradient direction
    return sig * (1 - sig)


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

# weights before 1.000.000 learning epochs
weights0 = np.array([
    [-0.424, -0.740, -0.961],
    [0.358, -0.893, -0.469],
    ])

weights1 = np.array([[-0.017], [-0.893], [0.148]])

# weights after 1.000.000 learning epochs with 99% average accuracy
# (0.006920605401532326 absolute_error_mean)
new_weights0 = np.array([
    [-13.12185181,  -1.02832185,   5.95205043],
    [5.95404295,  -1.02821446, -13.11618268],
    ])

new_weights1 = np.array([
    [16.34428342], 
    [-42.44674413], 
    [16.34345593],
    ])


# automatic weights generator. 
aut_weiths0 = 2 * np.random((2, 3)) - 1
aut_weiths1 = 2 * np.random((3, 1)) - 1
# np.random((inputs or hidden_layers, hidden_layers or output_layer))
# We multiply by 2 and subtract by 1 to generate some negative weights.


epoch = 1000000
learning_rate = 0.3
momentum = 1

for j in range(epoch):
    input_layer = inputs
    synapse_sum0 = np.dot(input_layer, weights0)
    hidden_layer = sigmoid(synapse_sum0)

    synapse_sum1 = np.dot(hidden_layer, weights1)
    output_layer = sigmoid(synapse_sum1)

    output_layer_error = outputs - output_layer
    absolute_error_mean = np.mean(np.abs(output_layer_error))

    # gradient descent calculus (weight adjustment) ------------------------------
    output_derivative = derivativeSigmoid(output_layer)
    output_delta = output_layer_error * output_derivative

    # Since the format of the output delta matrix (4x1) and the format of the weight1 matrix(3x1) are incompatible for the dot product, it is necessary to obtain a transposed matrix of the weights1 to perform the calculation of the dot product.
    transposed_weights1_matrix = weights1.T
    output_delta_X_weights1 = output_delta.dot(transposed_weights1_matrix)
    hidden_layer_delta = output_delta_X_weights1 * derivativeSigmoid(hidden_layer)

    # Something similar happens here with the backpropagation process
    transposed_hidden_layer = hidden_layer.T
    new_weights1 = transposed_hidden_layer.dot(output_delta)
    weights1 = (weights1 * momentum) + (new_weights1 * learning_rate)

    transposed_input_layer = input_layer.T
    new_weights0 = transposed_input_layer.dot(hidden_layer_delta)
    weights0 = (weights0 * momentum) + (new_weights0 * learning_rate) 

print(absolute_error_mean)
print()
print(weights0)
print()
print(weights1)