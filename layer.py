import numpy as np
from numpy.lib.stride_tricks import as_strided
# from scipy.signal import correlate2d

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
            # He initialization
            self.weights = np.random.normal(
                0,                                   # 0 avg  
                np.sqrt(2/self.input_size),          # sqrt(2/input_size) std
                (self.input_size, self.output_size)  # [input,output]
            )

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


class Conv2d(Layer):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, padding: int = 0, stride: int = 1):
        super().__init__()
        # expected input dimensions:
        # (batch_size, channels, height, width)
        self.channels_in = channels_in  # number of input channels, 1 - grayscale, 3 - rgb, equal to l-1 channles out
        self.channels_out = channels_out  # number of kernels, also equal to output depth 
        self.kernel_size = kernel_size  # kernel height and width

        self.padding = padding
        self.stride = stride

        # kernel dims:
        # (number of kernels, kernel (=input) depth, height, width)
        self.kernels_shape = (self.channels_out, self.channels_in, self.kernel_size, self.kernel_size)

        # He init for kernels
        self.kernels = np.random.normal(
            0, 
            np.sqrt(2/(self.channels_in * self.kernel_size**2)),  # num of features in a kernel = depth * height * width
            self.kernels_shape
        )
        self.bias = np.zeros(self.channels_out)


    def get_striding_windows(self, input_padded, stride: int):
        batch_size, channels_in, height_pad, width_pad = input_padded.shape

        # calculate H and W out to create output later
        height_out = (height_pad - self.kernel_size) // stride + 1  # // => floor division
        width_out = (width_pad - self.kernel_size) // stride + 1


        # create striding windows - every possible kH x kW patch
        shape = (batch_size, channels_in, height_out, width_out, self.kernel_size, self.kernel_size)
        strides = (
            # first 4 strides - moving the sliding window accross the input
            input_padded.strides[0],         # batch strides
            input_padded.strides[1],         # c_in strides
            input_padded.strides[2]*stride,  # h_out (row) strides
            input_padded.strides[3]*stride,  # w_out (col) strides
            # last 2 strides - moving inside a patch (inside a receptive field)
            input_padded.strides[2],         # kernel h (row) strides
            input_padded.strides[3],         # kernel w (col) strides
        )

        windows = as_strided(input_padded, shape=shape, strides=strides)
        return windows
        

    def forward(self, input, training):

        # apply padding
        if self.padding > 0:
            input_padded = np.pad(
                input,
                ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),  # batch and channels no pad so 0,0
                mode='constant'  # pad with 0s
            )
        else:
            input_padded = input

        # save input for backprop if training
        if training:
            self.input = input_padded

        # striding trick + tensordot implementation
        # 1. create striding windows
        windows = self.get_striding_windows(input_padded, self.stride)

        # 2. tensordot + bias
        # windows: (batch, c_in, h_out, w_out, kH, kW), so axes 1, 4, 5
        # kernels: (c_out, c_in, kH, kW), so axes 1, 2, 3
        output = np.tensordot(windows, self.kernels, axes=([1,4,5], [1,2,3]))
        # this gives us shape (batch_size, h_out, w_out, c_out)
        # we need (batch_size, c_out, h_out, w_out), so transpose
        output = output.transpose(0, 3, 1, 2)
        # add bias
        output += self.bias[None, :, None, None]  # proper broadcasting to add to channels

        return output
        




    def calculate_gradients(self, output_grad):
        ...


class Dropout(Layer):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input: np.ndarray, training: bool):
        if training:
            p = self.dropout_rate
            # new mask generated each forward pass
            self.mask = np.random.binomial(1, 1-p, input.shape)
            # passes the input through the mask with / (1-p) activation scaling
            return np.multiply(input, self.mask) / (1-p)
        # inference: full network is used, no dropout
        else:
            return input

    def calculate_gradients(self, output_grad: np.ndarray):
        # only updating the neurons which were not dropped
        return np.multiply(output_grad, self.mask), {}







# old conv forward
# batch_size, channels_in, height_pad, width_pad = input_padded.shape

# # calculate H and W out to create output later
# height_out = (height_pad - self.kernel_size) // self.stride + 1  # // => floor division
# width_out = (width_pad - self.kernel_size) // self.stride + 1


# I. simple loop implementation, problem: no way to adjust stride
# output = np.zeros((batch_size, self.channels_out, height_out, width_out))
# output += self.bias[None, :, None, None]  # reshape biases to [1, c_out, 1, 1], so it's added properly

# for batch_element in batch_size:
#     for c_out in self.channels_out:
#         for c_in in self.channels_in:
#             output[batch_element, c_out] += correlate2d(input_padded[batch_element, c_in], self.kernels[c_out, c_in], mode='valid')
        