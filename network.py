import numpy as np
from layer import Layer
from loss import Loss
from solver import Solver
from data import Dataset
from typing import List


class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.loss = None
        self.solver = None
        self.learning_rate = None

    def compile(self, loss: Loss, solver: Solver):
        self.loss = loss
        self.solver = solver

    def forward(self, input: np.ndarray, training=True) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input, training=training)
        return input

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input, training=False)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        grad = self.loss.get_grad(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.solver)

    def train(self, dataset: Dataset, epochs: int):
        for epoch in range(1, epochs+1):
            total_epoch_loss = 0

            for x_train, y_train in dataset:  # same as for batch in dataset:
                # forward pass
                y_pred = self.forward(x_train)

                # calculate loss
                total_epoch_loss += np.sum(self.loss.get_loss(y_train, y_pred, vectorized=True))  # sum of all the errors from each observation 

                # backward pass
                self.backward(y_train, y_pred)

            avg_epoch_loss = total_epoch_loss / len(dataset)  # works even for smaller last batch

            print(f"Epoch {epoch}/{epochs} - Loss: {avg_epoch_loss:.4f}")


# TODO shuffle & batch
# TODO store loss for plots

