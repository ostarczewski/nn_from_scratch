import numpy as np
from typing import Tuple

# TODO SGD with momentum
# SGD with momentum will match each layer with a saved set of stored values like momentum 

class Solver:
    def __init__(self, lr: float):
        self.lr = lr

    def layer_update(self, input: np.ndarray, weights: np.ndarray, bias: np.ndarray, output_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class SGD(Solver):
    def __init__(self, lr: float):
        super().__init__(lr)
            
    def layer_update(self, input: np.ndarray, weights: np.ndarray, bias: np.ndarray, output_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # L wrt W = L wrt Z * Z wrt W
        # Z = A-1 * W + B (forward pass formula)
        # Z wrt W = A-1 (input)
        # L wrt W = grad l * A-1 (output grad * input)

        # output grad -> batch x output
        # w           -> input x output
        # input       -> batch x input
        # input x batch * batch x ouput <= input.T * output grad

        weights_grad = np.dot(input.T, output_grad) / input.shape[0]
        # dot product sums the impact of each individual obs, so we need to divide the matrix by batch size


        # b = b - lr * L wrt B
        # Z wrt b = 1
        # L wrt b = grad l

        bias_grad = np.mean(output_grad, axis=0)
        # avg grad of each bias over all observation in the batch

        updated_weights = weights - self.lr * weights_grad
        updated_bias = bias - self.lr * bias_grad
        return updated_weights, updated_bias

