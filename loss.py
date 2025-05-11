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
        def mse(y_true: np.ndarray, y_pred: np.ndarray, vectorized: bool = False) -> Union[float, np.ndarray]:
            if vectorized:
                mean_dimensions = tuple(range(1, len(y_true.shape)))  # np (1, 2, 3) jezeli dane sa wysokowymiarowe, to musimy zrobic srednia po wszystkich wymiarach (oprócz rzędów)
                mse_vector = np.mean((y_true - y_pred)**2, axis=mean_dimensions)  # średnia po wierszach
                return np.reshape(mse_vector, (-1,1))  # zmiana na vector kolumnowy
            else:
                return np.mean((y_true - y_pred)**2)

        # MSE loss function derivative with respect to predicted vaules y^
        def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            n = y_true.shape[0]
            return -2/n * (y_true - y_pred)
        
        super().__init__(mse, mse_prime)



"""
wytłumaczenie MSE prime
d (y-y^)^2 / d y^

chain rule
d (y-y^)^2 / d y^ = d (y-y^)^2 / d y-y^  *  d y-y^ / d y^

= 2*(y-y^) * -1    (bo kwadrat schodzi do 2, a w drugim y=>0, y^=>1)

więc d mse / d y^ 
= -2*(y-y^) / n   <- gradient per yi, nie całkowity

"""


# TODO log loss & binary log loss

def cross_entropy_loss(y_true, y_pred):
    n = y_true.shape[0]
    return -sum(y_true * np.log(y_pred + 1e-9)) / n  # epsilon to avoid 0 log, div by N to account for batch size <- this will be handled in the network code
