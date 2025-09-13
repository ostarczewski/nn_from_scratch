import numpy as np
from typing import Union


class Loss:
    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def get_loss(self, y_true: np.ndarray, y_pred: np.ndarray, vectorized: bool = False) -> Union[float, np.ndarray]:
        return self.loss(y_true, y_pred, vectorized)

    def get_grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.loss_prime(y_true, y_pred)
    

class MeanSquaredError(Loss):
    def __init__(self):
        def forward(y_true: np.ndarray, y_pred: np.ndarray, vectorized: bool = False) -> Union[float, np.ndarray]:
            if vectorized:
                mse_vector = np.mean((y_true - y_pred)**2, axis=1)  # średnia po wierszach
                return np.reshape(mse_vector, (-1,1))  # zmiana na vector kolumnowy
            else:
                return np.mean((y_true - y_pred)**2)

        # MSE loss function derivative with respect to predicted vaules y^
        def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            n = y_true.shape[0]
            # normalizing over batch size in loss, so later we sum the gradients 
            return -2/n * (y_true - y_pred)
        
        super().__init__(forward, backward)


class CrossEntropyLoss(Loss):
    def __init__(self):
        # forward in 2 stages -> softmax, cross enrtopy
        def softmax(input):
            # numerical stability trick with -max
            exponents = np.exp(input - np.max(input, axis=1, keepdims=True))  # max po wierszach, keep dims for proper broadcasting
            # returns a matrix
            return exponents / np.sum(exponents, axis=1, keepdims=True)
        
        def forward(y_true: np.ndarray, y_pred: np.ndarray, vectorized: bool = False) -> Union[float, np.ndarray]:
            # softmax the values to get actual y pred
            y_pred = softmax(y_pred)
            # epsilon needed so no log(0) for any class
            epsilon = 1e-12
            # calculate loss vector
            loss_vector = -np.sum(y_true * np.log(y_pred+epsilon), axis=1)  # one value per row

            if vectorized:
                return np.reshape(loss_vector, (-1,1))  # column vector
            else:
                return np.mean(loss_vector)
            
        def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            # softmax values
            y_pred = softmax(y_pred)
            n = y_true.shape[0]
            # loss grad normalized by batch size, returns a matrix
            return (y_pred - y_true) / n
        
        super().__init__(forward, backward)
    



# MSE prime:
# d (y-y^)^2 / d y^

# chain rule
# d (y-y^)^2 / d y^ = d (y-y^)^2 / d y-y^  *  d y-y^ / d y^
# = 2*(y-y^) * -1

# = -2*(y-y^) / n   <- gradient per yi
