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

    def forward(self, input: np.ndarray, training=True) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input, training=training)
        return input

    def predict(self, input: np.ndarray) -> np.ndarray:
        self.forward(input, training=False)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        grad = self.loss.get_grad(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate, self.solver)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 0, shuffle: bool = False):
        # avoiding different behavior for batch and non-batch training
        if not 0 < batch_size <= x_train.shape[0]:
            batch_size = x_train.shape[0]  # no batching
        
        
        for epoch in range(1, epochs+1):
            # local x and y train for shuffle and batch operations
            # data = data_tools.dataset(x_train, y_train)
            data = [{'x_train': x_train, 'y_train': y_train}]  # rozwiązanie tymczasowe

            # if shuffle:
                # data = data_tools.shuffle(data) 
                # to be implemented

            # always batching
            # data = data_tools.batch(data, batch_size, drop_last=...)
            # when batch = len(x_train) then [data] turns into [[data]]

            total_epoch_loss = 0

            for batch in data:
                # forward pass
                y_pred = self.forward(batch['x_train'])

                # calculate loss
                total_epoch_loss += np.sum(self.loss.get_loss(batch['y_train'], y_pred, vectorized=True))  # sum of all the errors from each observation 

                # backward pass
                self.backward(batch['y_train'], y_pred, learning_rate)

            avg_epoch_loss = total_epoch_loss / x_train.shape[0]  # works even for smaller last batch

            print(f"Epoch {epoch}/{epochs} - Loss: {avg_epoch_loss:.4f}")


# TODO shuffle & batch
# TODO store loss for plots

# TODO forward different for training and inference (dropout, later maybe batch norm for cnn)

# pip install cupy-cuda12x
