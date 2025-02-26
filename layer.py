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
        # self.weights = np.random.randn(input_size, output_size)                               # random initialization
        self.weights = np.random.normal(0, np.sqrt(2/input_size), (input_size, output_size))  # He initialization
        # N ~ zero avg, sqrt(2/input_size) std
        self.bias = np.zeros(output_size)
        # self.bias = np.full(output_size, 0.01)  # można dać 0.01 dla ReLU


    def forward(self, input):
        self.input = input  # zapisywanie inputu, po co? do back prop?
        self.output = self.input @ self.weights + self.bias
        # [1,input] @ [input,output] + [1,output]  =>  [1,output]

    def backward(self, output_grad, learning_rate):
        raise NotImplementedError("Not implemented yet")