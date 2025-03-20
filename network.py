import numpy as np
from layer import Layer
from loss import Loss
from typing import List

class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.loss = None
        self.solver = None
        self.learning_rate = None

    def compile(self, loss: Loss, solver):
        self.loss = loss
        self.solver = solver

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, y_true, y_pred, learning_rate):
        grad = self.loss.get_grad(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate, self.solver)
