import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# MSE loss function derivative with respect to predicted vaules y^
def mse_prime(y_true, y_pred):
    n = len(y_true)  # lub np.size() lub .shape[0]
    return -2/n * (y_true - y_pred)
    # gradients for backprop are calculated per yi, so we need a vector



"""
czyli
d (y-y^)^2 / d y^

chain rule
d (y-y^)^2 / d y^ = d (y-y^)^2 / d y-y^  *  d y-y^ / d y^

= 2*(y-y^) * -1    (bo kwadrat schodzi do 2, a w drugim y=>0, y^=>1)

więc d mse / d y^ 
= -2*(y-y^) / n   <- gradient per yi, nie całkowity

"""

def cross_entropy_loss(y_true, y_pred):
    n = len(y_true)
    return -sum(y_true * np.log(y_pred + 1e-9)) / n  # epsilon to avoid 0 log, div by N to account for batch size
