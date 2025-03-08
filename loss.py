import numpy as np


class Loss:
    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def get_loss(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def get_grad(self, y_true, y_pred):
        return self.loss_prime(y_true, y_pred)
    

class MeanSquaredError(Loss):
    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred)**2)

        # MSE loss function derivative with respect to predicted vaules y^
        def mse_prime(y_true, y_pred):
            n = np.size(y_true)
            return -2/n * (y_true - y_pred)
        
        super().__init__(mse, mse_prime)



    # gradients for backprop are calculated per yi, so we need a vector



"""
wytłumaczenie MSE prime
d (y-y^)^2 / d y^

chain rule
d (y-y^)^2 / d y^ = d (y-y^)^2 / d y-y^  *  d y-y^ / d y^

= 2*(y-y^) * -1    (bo kwadrat schodzi do 2, a w drugim y=>0, y^=>1)

więc d mse / d y^ 
= -2*(y-y^) / n   <- gradient per yi, nie całkowity

"""

def cross_entropy_loss(y_true, y_pred):
    n = np.size(y_true)
    return -sum(y_true * np.log(y_pred + 1e-9)) / n  # epsilon to avoid 0 log, div by N to account for batch size
