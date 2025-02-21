import numpy as np

# base layer to later inherit from

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, output_grad, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        # tutaj kolejność zmienia to czy 1 neuron to kolumna czy rząd
        # możliwe, że będzie błąd w mnożeniu przez różne wielkości macierzy jak się to na odwrót zrobi
        # self.weights = np.random.randn(input_size, output_size) 
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros(output_size)
        # self.bias = np.random.randn(output_size)

        # considerations:
        # Small weights pevent exploding/vanishing gradients
        # Bias is initialized to zero because it doesn’t need to be random—it shifts the output, and training will adjust it.

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias

    def backward(self, output_grad, learning_rate):
        # olala
        # raise NotImplementedError("Not implemented yet")
        # tutaj znowu kolejność ma znaczenie
        # weights_grad = np.dot(self.input.T, output_grad)   ???
        weights_grad = np.dot(output_grad, self.input.T) 


        # ja muszę sobie to wszystko zrozumieć na kartce, a potem mogę to zaimplementować