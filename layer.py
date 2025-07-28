import numpy as np

class Layer:
    def __init__(self):
        self.input = None
    
    def forward(self, input: np.ndarray, training: bool):
        pass

    def calculate_gradients(self, output_grad: np.ndarray):
        pass


class Dense(Layer):
    def __init__(self, output_size: int, input_size: int = None):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size

        self.bias = np.zeros(self.output_size)
        self.weights = None


    def forward(self, input: np.ndarray, training: bool):
        # initialize weights if None
        if self.weights is None:
            if not self.input_size:
                self.input_size = input.shape[1]
            # He initialization, N ~ zero avg, sqrt(2/input_size) std
            self.weights = np.random.normal(0, np.sqrt(2/self.input_size), (self.input_size, self.output_size))

        # store input for backprop when training
        if training:
            self.input = input 
        return np.dot(input, self.weights) + self.bias
        # [batch,input] @ [input,output] + [1,output]  =>  [batch,output]


    def calculate_gradients(self, output_grad: np.ndarray):
        # w l * ouput grad calculation to pass the gradient further
        input_grad = np.dot(output_grad, self.weights.T)

        # dot product sums the impact of each individual obs, so we need to divide the matrix by batch size
        weights_grad = np.dot(self.input.T, output_grad) / self.input.shape[0]
        
        # avg grad of each bias over all observation in the batch
        bias_grad = np.mean(output_grad, axis=0)
        
        # returns a dict, where key is the param name to be updated by solver
        param_grad = {"weights": weights_grad, "bias": bias_grad}

        # passing the input grad backwards to the preceding layer (and returning the grads needed for param updates)
        return input_grad, param_grad


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
        # inference: full network is used, no dropout
        else:
            return input

    def calculate_gradients(self, output_grad: np.ndarray):
        return np.multiply(output_grad, self.mask), {}

