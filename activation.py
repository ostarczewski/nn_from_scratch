import numpy as np
from layer import Layer

# TODO parametrical relu, def backward będzie w nim do zmiany?

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_grad, learning_rate):
        activation_grad = self.activation_prime(self.input)
        return np.multiply(output_grad, activation_grad)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, 1, 0)
        
        super().__init__(relu, relu_prime)


class LeakyReLU(Activation):
    def __init__(self, slope = 0.01):
        self.slope = slope

        def leaky_relu(x):
            return np.maximum(0, x) + self.slope * np.minimum(0, x)

        def leaky_relu_prime(x):
            return np.where(x > 0, 1, self.slope)
        
        super().__init__(leaky_relu, leaky_relu_prime)

