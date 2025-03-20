import numpy as np

# TODO network to powinien byc obiekt, ktory tworzymy
# dajemy mu parametry sieci, optimiser, 
# weight init metod i bias init valule jako opcjonalne

class Layer:
    def __init__(self):
        # po co to wgl jest?
        self.input = None
        self.output = None
    
    # forward backward here just to create the structure
    def forward(self, input):
        pass

    def backward(self, output_grad, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        # super().__init__()   <-- i guess nie potrzebne?
        # self.weights = np.random.randn(input_size, output_size)                               # random initialization
        self.weights = np.random.normal(0, np.sqrt(2/input_size), (input_size, output_size))    # He initialization
        # N ~ zero avg, sqrt(2/input_size) std
        self.bias = np.zeros(output_size)
        # self.bias = np.full(output_size, 0.01)  # można dać 0.01 dla ReLU

    def forward(self, input):
        self.input = input 
        return self.input @ self.weights + self.bias
        # [batch,input] @ [input,output] + [1,output]  =>  [batch,output]

    def backward(self, output_grad, learning_rate):
        raise NotImplementedError("Not implemented yet")
        # jeżeli chcemy mieć możliwość wyboru optimisera, to trzeba będzie w jakiś sposób powiedzieć o tym każdemu layerowi


# pip install cupy-cuda12x