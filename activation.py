import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self):
        super().__init__()
        # self.activation = activation
        # self.activation_prime = activation_prime

    def forward(self, input, training):
        if training:
            self.input = input 
        return self.activation(input)
    
    def calculate_gradients(self, output_grad: np.ndarray):
        activation_grad = self.activation_prime(self.input)
        return np.multiply(output_grad, activation_grad), {}


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def activation(self, x):
        return np.maximum(0, x)

    def activation_prime(self, x):
        return np.where(x > 0, 1, 0)


class LeakyReLU(Activation):
    def __init__(self, slope=0.01):
        super().__init__()
        self.slope = slope

    def activation(self, x):
        return np.where(x > 0, x, self.slope * x)

    def activation_prime(self, x):
        return np.where(x > 0, 1, self.slope)


# class ReLU(Activation):
#     def __init__(self):
#         def relu(x):
#             return np.maximum(0, x)

#         def relu_prime(x):
#             return np.where(x > 0, 1, 0)
        
#         super().__init__(relu, relu_prime)


# class LeakyReLU(Activation):
#     def __init__(self, slope = 0.01):
#         self.slope = slope

#         def leaky_relu(x):
#             return self.slope * np.minimum(0, x) + np.maximum(0, x)

#         def leaky_relu_prime(x):
#             return np.where(x > 0, 1, self.slope)
        
#         super().__init__(leaky_relu, leaky_relu_prime)




# TODO argument if one value should be used across all channels?
# będą komplikacje z tym 

class PReLU(Activation):
    def __init__(self, init_slope = 0.25):
        self.init_slope = init_slope
        self.alfas = None

    def forward(self, input, training):
        if self.alfas is None:
            # auto input size detection
            self.alfas = np.full(input.shape[1], self.init_slope)

        if training:
            self.input = input
        return np.where(input >= 0, input, input * self.alfas)
    
    def calculate_gradients(self, output_grad: np.ndarray):
        # activation grad w przypadku jednego slope'u
        # activation_grad = np.where(self.input > 0, 1, self.slope)
        # return np.multiply(output_grad, activation_grad), {}

        # input grad to pass on - later
        # prelu_derivative = np.where(self.input > 0, 1, self.slope)
        # potem to
        # return np.multiply(output_grad, prelu_derivative), {}

        # slope_grad = ...

        pass




# TODO parametric relu, def backward będzie w nim do zmiany?
# żeby PReLU działało taka warstwa będzie musiała mieć serię wag a, które będą modyfikowalne przez back prop
# czyli zamiast self.slope będzie self.params = np.array([0.01, 0.02, 0.01, ...])
# i forward będzie musiało przepuszczać każdy odpowiedni input przez każde odpowiednie PReLU
