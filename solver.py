import numpy as np
from typing import Tuple

# TODO SGD with momentum
# SGD with momentum will match each layer with a saved set of stored values like momentum 

class Solver:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, input: np.ndarray, weights: np.ndarray, bias: np.ndarray, output_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class SGD(Solver):
    def __init__(self, lr: float):
        super().__init__(lr)
            
    def step(self, input: np.ndarray, weights: np.ndarray, bias: np.ndarray, output_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        weights_grad = np.dot(input.T, output_grad)
        bias_grad = output_grad

        updated_weights = weights - self.lr * weights_grad
        updated_bias = bias - self.lr * bias_grad
        return updated_weights, updated_bias



# def SGD(input: np.ndarray, weights: np.ndarray, bias: np.ndarray, learning_rate: float, output_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         # we need to calculate weights_grad
#         # output grad = grad l
#         # w = w - lr * dL wrt dW
#         # L wrt W = L wrt Z * Z wrt W
#         # Z = A-1 * W + B
#         # Z wrt W = A-1
#         # L wrt W = grad l * A-1

#         # output grad -> batch x output
#         # w           -> input x output
#         # a-1 = input -> batch x input
#         # a-1.T * output grad
#         # input x batch * batch x ouput -> input x output
#     weights_grad = np.dot(input.T, output_grad)


#         # b = b - lr * L wrt B
#         # Z wrt b = 1
#         # L wrt b = grad l
#     bias_grad = output_grad

#     updated_weights = weights - learning_rate * weights_grad
#     updated_bias = bias - learning_rate * bias_grad
#     return updated_weights, updated_bias


