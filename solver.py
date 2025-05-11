import numpy as np
from typing import Tuple

# TODO solver as a class
# solver will be initialized in the network, and passed to the layers
# it'll match each layer with a saved set of stored values like momentum 

def SGD(input: np.ndarray, weights: np.ndarray, bias: np.ndarray, learning_rate: float, output_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # we need to calculate weights_grad
    # output grad = grad l
    # w = w - lr * dL wrt dW
    # L wrt W = L wrt Z * Z wrt W
    # Z = A-1 * W + B
    # Z wrt W = A-1
    # L wrt W = grad l * A-1

    # output grad -> batch x output
    # w           -> input x output
    # a-1 = input -> batch x input
    # a-1.T * output grad
    # input x batch * batch x ouput -> input x output
    weights_grad = np.dot(input.T, output_grad)


    # b = b - lr * L wrt B
    # Z wrt b = 1
    # L wrt b = grad l
    bias_grad = output_grad

    updated_weights = weights - learning_rate * weights_grad
    updated_bias = bias - learning_rate * bias_grad
    return updated_weights, updated_bias

# TODO SGD with momentum
