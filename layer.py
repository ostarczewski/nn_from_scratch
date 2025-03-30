import numpy as np

# TODO
# weight init metod i bias init valule jako opcjonalne, ale to chyba już w Network

class Layer:
    def __init__(self):
        # po co to wgl jest?
        self.input = None
        self.output = None
    
    # forward backward here just to create the structure
    def forward(self, input):
        pass

    def backward(self, output_grad, learning_rate, solver):
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
        return np.dot(self.input, self.weights) + self.bias
        # [batch,input] @ [input,output] + [1,output]  =>  [batch,output]

    def backward(self, output_grad, learning_rate, solver):
        # output grad = grad l
        # input grad = w l * grad l
        # grad l-1 = f'(z l-1) * w l * grad l

        # output grad -> batch x output
        # input grad  -> batch x input
        # batch x input = batch x output * ouput x input -> w.T
        
        # param update
        self.weights, self.bias = solver(self.input, self.weights, self.bias, learning_rate, output_grad)

        # w l * ouput grad calculation to pass the gradient further
        input_grad = np.dot(output_grad, self.weights.T)
        return input_grad

# pip install cupy-cuda12x