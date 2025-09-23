import numpy as np
from numpy.lib.stride_tricks import as_strided

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

        self.bias = np.zeros(self.output_size, dtype=np.float32)
        self.weights = None


    def forward(self, input: np.ndarray, training: bool):
        # initialize weights if None
        if self.weights is None:
            if not self.input_size:
                self.input_size = input.shape[1]
            # He initialization
            self.weights = np.random.normal(
                0,                                    # 0 avg  
                np.sqrt(2/self.input_size),           # sqrt(2/input_size) std
                (self.input_size, self.output_size),  # [input,output]
            ).astype(np.float32)

        # store input for backprop when training
        if training:
            self.input = input 
        return np.dot(input, self.weights) + self.bias
        # [batch,input] @ [input,output] + [1,output]  =>  [batch,output]


    def calculate_gradients(self, output_grad: np.ndarray):
        # w l * ouput grad calculation to pass the gradient further
        input_grad = np.dot(output_grad, self.weights.T)

        # loss derivative already divided grad by batch size
        weights_grad = np.dot(self.input.T, output_grad)
        
        # sum the contributions of each obs in the batch for each bias
        bias_grad = np.sum(output_grad, axis=0)
        
        # returns a dict, where key is the param name to be updated by solver
        param_grad = {"weights": weights_grad, "bias": bias_grad}

        # passing the input grad backwards to the preceding layer (and returning the grads needed for param updates)
        return input_grad, param_grad



class Dropout(Layer):
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input: np.ndarray, training: bool):
        if training:
            p = np.float32(self.dropout_rate)
            # new mask generated each forward pass
            self.mask = np.random.binomial(1, 1-p, input.shape).astype(np.float32)
            # passes the input through the mask with / (1-p) activation scaling
            return np.multiply(input, self.mask) / (1-p)
        # inference: full network is used, no dropout
        else:
            return input

    def calculate_gradients(self, output_grad: np.ndarray):
        # only updating the neurons which were not dropped
        return np.multiply(output_grad, self.mask), {}



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
        ).astype(np.float32)

        self.bias = np.zeros(self.channels_out).astype(np.float32)


    def get_striding_windows(self, input: np.ndarray, stride: int):
        batch_size, channels_in, height, width = input.shape

        # calculate H and W out to create output later
        height_out = (height - self.kernel_size) // stride + 1  # // => floor division
        width_out = (width - self.kernel_size) // stride + 1


        # create striding windows - every possible kH x kW patch
        shape = (batch_size, channels_in, height_out, width_out, self.kernel_size, self.kernel_size)
        strides = (
            # first 4 strides - moving the sliding window accross the input
            input.strides[0],         # batch strides
            input.strides[1],         # c_in strides
            input.strides[2]*stride,  # h_out (row) strides
            input.strides[3]*stride,  # w_out (col) strides
            # last 2 strides - moving inside a patch (inside a receptive field)
            input.strides[2],         # kernel h (row) strides
            input.strides[3],         # kernel w (col) strides
        )

        windows = as_strided(input, shape=shape, strides=strides)
        return windows
        

    def forward(self, input: np.ndarray, training: bool):
        # apply padding
        if self.padding > 0:
            input = np.pad(
                input,
                ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),  # batch and channels no pad so 0,0
                mode='constant'  # pad with 0s
            )

        # save (padded) input for backprop if training
        if training:
            self.input = input

        # striding trick + tensordot implementation
        # 1. create striding windows
        windows = self.get_striding_windows(input, self.stride)

        # 2. tensordot + bias
        # windows: (batch, c_in, h_out, w_out, kH, kW), so axes 1, 4, 5
        # kernels: (c_out, c_in, kH, kW), so axes 1, 2, 3
        output = np.tensordot(windows, self.kernels, axes=([1,4,5], [1,2,3]))
        # this gives us shape (batch_size, h_out, w_out, c_out)
        # we need (batch_size, c_out, h_out, w_out), so transpose
        output = output.transpose(0, 3, 1, 2)
        # add bias
        output += self.bias[None, :, None, None]  # proper broadcasting to add bias to all neurons per channel

        return output


    def calculate_gradients(self, output_grad: np.ndarray):
        # output grad shape: (batch, c_out, h_out, w_out)
        input = self.input

        # bias grad sums the output grad across the channel (dim 1 -> c_out)
        bias_grad = output_grad.sum((0, 2, 3))
        
        # kernel (weights) grad - same as in dense layer, an operation between layer input and output_grad
        # but instead of dot product we do cross corr!
        input_windows = self.get_striding_windows(input, self.stride)
        # windows shape: (batch, c_in, h_out, w_out, kH, kW)
        # output grad shape: (batch, c_out, h_out, w_out)
        kernels_grad = np.tensordot(input_windows, output_grad, axes=([0,2,3], [0,2,3]))
        # shape: (c_in, kH, kW, c_out)
        kernels_grad = kernels_grad.transpose(3, 0, 1, 2)
        # now shape: (c_out, c_in, kH, kW), matches kernels shape


        # input grad
        kernels_flipped = np.flip(self.kernels, axis=(2,3))  # kernels shape (c_out, c_in, kH, kW), so kH kW

        # inserting 0s between output grad (upsampling) to account for stride
        # calculate the size of the new output grad matrix
        if self.stride > 1:
            rows, cols = output_grad.shape[-2:]  # two last dimensions
            filled_rows = rows + (rows - 1) * (self.stride - 1)
            filled_cols = cols + (cols - 1) * (self.stride - 1)
            filled_output_grad = np.zeros((output_grad.shape[0], output_grad.shape[1], filled_rows, filled_cols))
            filled_output_grad[:, :, ::self.stride, ::self.stride] = output_grad
        else:
            filled_output_grad = output_grad

        # when calculating input_grad, we always pad by k_size-1 allowing for a full convoluction
        out_grad_pad = self.kernel_size - 1
        output_grad_padded = np.pad(
            filled_output_grad, 
            ((0,0), (0,0), (out_grad_pad, out_grad_pad), (out_grad_pad, out_grad_pad)),
            mode='constant'
        )

        output_grad_windows = self.get_striding_windows(output_grad_padded, stride=1)
        # shape: (batch, c_out, h_in_pad, w_in_pad, kH, kW))
        # kernels fliped shape: (c_out, c_in, kH, kW)
        input_grad = np.tensordot(output_grad_windows, kernels_flipped, axes=([1,4,5], [0,2,3]))
        # this gives us shape (batch, h_in_pad, w_in_pad, c_in)
        # we need (batch, c_in, h_in, w_in) -> transpose + remove padding to match original input shape
        input_grad = input_grad.transpose(0, 3, 1, 2)

        # unpad the input grad if padding applied in forward pass
        if self.padding > 0:
            input_grad = input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]


        # dict for solver, key = param name
        param_grad = {"kernels": kernels_grad, "bias": bias_grad}
        # pass input grad back to l-1, give param grad for solver
        return input_grad, param_grad



class MaxPool2d(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: np.ndarray, training: bool):
        batch_size, channels, height, width = input.shape

        # calculate H and W out to create output later
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
    
        # we split the input into kH x kW blocks
        # example: if k = 2, each 2x2 patch becomes a [x0, x1, x2, x3] vector
        input_reshaped = input.reshape(batch_size, channels, 
                                       height//self.kernel_size, self.kernel_size, 
                                       width//self.kernel_size, self.kernel_size)
        # we get shape (batch, channels, vertical patches num, vert patch height, horizontal patch num, horiz patch width)
        input_reshaped = input_reshaped.swapaxes(3,4)  # swap so patch num first, patch height/width 2nd
        
        input_reshaped = input_reshaped.reshape(batch_size, channels, height_out*width_out, self.kernel_size**2)
        # shape (batch, channels, total num of patches, patch size flattened)
        # now we need a max value from each patch
        output = np.max(input_reshaped, axis=3)
        # shape (batch, channels, patch_num), for each patch one max val

        # what we need for backward:
        # input shape
        # output h, w
        # max ids, to know which elements the gradients should be passed back to
        if training:
            self.input_shape = input.shape
            self.h_out, self.w_out = height_out, width_out
            # max ids -> 1 row consists n numbers, where n is number of patches extracted from one channel per one batch item
            # each number tells us the highest value id in a patch
            self.max_ids = np.argmax(input_reshaped, axis=3)  # gives us a pointer for passing back the gradients
            # shape (batch, channels, patch_num)

        # unflatten the max values from patches
        return output.reshape(batch_size, channels, height_out, width_out)
    

    def calculate_gradients(self, output_grad: np.ndarray):
        batch_size, channels, height, width = self.input_shape
        # flatten max_ids
        max_ids = self.max_ids.reshape(-1)
        
        # initial input grad filled with 0s, those 0s will be modified for max activation neurons
        # we also need height and width in long 1d format so we can access ids later
        input_grad_long = np.zeros(self.input_shape, dtype=np.float32).reshape(batch_size, channels, height*width)

        # each patch gets a batch id asigned
        # the batch num is repeated channels * H_out * W_out times, bc that's the amount of patches in a batch
        batch_idx = np.repeat(np.arange(batch_size), channels*self.h_out*self.w_out)
        # channel id gets repeated H_out * W_out times -> amount of patches in a channel
        # then this pattern is the same across all batches -> np.tile
        channel_idx = np.tile(np.repeat(np.arange(channels), self.h_out*self.w_out), batch_size)
        # each patch per batch/channel combination gets an id
        # so for each batch/channel combination, the pattern 1, ..., P is repeated, where P is the last patch
        patch_idx = np.tile(np.arange(self.h_out*self.w_out), batch_size*channels)

        # now we calculate which row and column the max obs was in in the original input
        # row calculation
        # gets the row starting position of each patch
        start_row_idx = (patch_idx // self.w_out) * self.stride
        # adds the information in which row the max activation was for each patch
        max_row_idx = start_row_idx + (max_ids // self.kernel_size)
        # column calculation
        # we use modulo, so that if ouput was 3x3, the first column in patch is on id 0, 3, 6... 
        start_col_idx = (patch_idx % self.w_out) * self.stride
        # adds the info about which column the max activation was actually in for each patch
        max_col_idx = start_col_idx + (max_ids % self.kernel_size)

        # for each patch we get the actual pixel id with the highest activation
        # max_row_idx * width moves to the start of the row when we're in 1D vector
        # + max_col_idx moves to the exact column in the row
        input_max_ids = max_row_idx * width + max_col_idx

        # now each max activation neuron gets assigned a proper gradient, per patch/channel/batch
        # every output_grad value (or grad for max value for each patch)
        # gets assigned a proper batch, channel, and place in the patch which max activation was in
        input_grad_long[batch_idx, channel_idx, input_max_ids] = output_grad.reshape(-1)

        # reshape to the original shape, H * W are now in wide format (2d) not long (1d)
        return input_grad_long.reshape(self.input_shape), {}



class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, training: bool):
        # input shape; (batch, c, h, w)
        if training:
            self.input_shape = input.shape
        # flatten
        return input.reshape(input.shape[0], -1)  # flatten per each observation

    def calculate_gradients(self, output_grad: np.ndarray):
        return output_grad.reshape(self.input_shape), {}

