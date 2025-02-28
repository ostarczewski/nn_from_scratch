import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# MSE loss function derivative with respect to predicted vaules y^
def mse_prime(y_true, y_pred):
    n = len(y_pred)
    return -2/n * sum(y_true - y_pred)

    # lub 2/n * np.sum(y_pred - y_true), na odwrót


"""
czyli
d (y-y^)^2 / d y^

chain rule
d (y-y^)^2 / d y^ = d (y-y^)^2 / d y-y^  *  d y-y^ / d y^

= 2*(y-y^) * -1    (bo kwadrat schodzi do 2, a w drugim y=>0, y^=>1)

gdy dodamy do tego sumę i dzielenie przez n, to mamy
-2/n * sum(y-y^)
lub
sum(2(y-y^) * -1) / n

"""

