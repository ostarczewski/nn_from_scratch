import numpy as np

# TODO
# weight init metod i bias init valule jako opcjonalne (maybe)

class Layer:
    def __init__(self):
        self.input = None
        # self.output = None  wer're not storing the outputs anyway for now
    
    def forward(self, input: np.ndarray, training: bool):
        pass

    def backward(self, output_grad: np.ndarray, learning_rate: float):
        pass


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # self.weights = np.random.randn(input_size, output_size)                               # random initialization
        self.weights = np.random.normal(0, np.sqrt(2/input_size), (input_size, output_size))    # He initialization
        # N ~ zero avg, sqrt(2/input_size) std
        self.bias = np.zeros(output_size)
        # self.bias = np.full(output_size, 0.01)  # można dać 0.01 dla ReLU

    def forward(self, input: np.ndarray, training: bool):
        if training:
            self.input = input 
        return np.dot(input, self.weights) + self.bias
        # [batch,input] @ [input,output] + [1,output]  =>  [batch,output]

    def backward(self, output_grad: np.ndarray, learning_rate: float, solver):
        # output grad = grad l
        # input grad = w l * grad l
        # grad l-1 = f'(z l-1) * w l * grad l

        # output grad -> batch x output
        # input grad  -> batch x input
        # batch x input = batch x output * ouput x input -> w.T

        # w l * ouput grad calculation to pass the gradient further
        input_grad = np.dot(output_grad, self.weights.T)
        
        # param update
        self.weights, self.bias = solver(self.input, self.weights, self.bias, learning_rate, output_grad)
        
        return input_grad


class Dropout(Layer):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input: np.ndarray, training: bool):
        if training:
            p = self.dropout_rate
            self.mask = np.random.binomial(1, 1-p, input.shape)
            # later can use .astype(input.dtype) with the matrix when dealing with float precission
            return np.multiply(input, self.mask) / (1-p)  # activation scaling
        else:
            return input

    def backward(self, output_grad: np.ndarray, learning_rate: float):
        return np.multiply(output_grad, self.mask)
    
    # TODO maybe set a random seed?
