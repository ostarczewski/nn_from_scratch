import numpy as np
from layer import Layer
from typing import Tuple

# TODO SGD with momentum
# SGD with momentum will match each layer with a saved set of stored values like momentum 

class Solver:
    def __init__(self, lr: float):
        self.lr = lr

    def apply_gradients(self, layer: Layer, param_grad: dict):
        pass


class SGD(Solver):
    def __init__(self, lr: float):
        super().__init__(lr)
            
    def apply_gradients(self, layer: Layer, param_grad: dict):
        # layer will have a empty dict if no params to update
        if param_grad:
            # for each param (weights, biases)
            for param_name, grad in param_grad.items():
                # access the parameter value (like info about weights)
                param = getattr(layer, param_name)

                # update rule for SGD
                updated_param = param - self.lr * grad 

                # update param vaues
                setattr(layer, param_name, updated_param)

