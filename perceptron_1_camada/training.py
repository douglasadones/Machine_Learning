import numpy as np

inputs = np.array([[0,0], [0,1], [1, 0], [1,1]])
outputs = np.array([0, 0, 0, 1])
weights = np.array([0.0, 0.0])
learning_rate = 0.1

def stepFunction(sum):
    if sum >= 1:
        return 1
    return 0


def outputCalculate(register):
    s = register.dot(weights)
    return stepFunction(s)


def training():
    total_error = 1
    while total_error != 0:
        total_error = 0
        for i in range(len(outputs)):
            calculed_output = outputCalculate(np.array(inputs[i]))
            error = abs(outputs[i] - calculed_output)
            total_error += error
            for j in range(len(weights)):
                weights[j] = weights[j] + (learning_rate * inputs[i][j] * error)
                print(f'Updated Weight: {weights[j]}')
        print(f'Total error: {total_error}')


training()
