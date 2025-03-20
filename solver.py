import numpy as np

def SGD(input, weights, bias, learning_rate, output_grad):
    # we need to calculate weights_grad
    # output grad = grad l
    # w = w - lr * dL wrt dW
    # L wrt W = L wrt Z * Z wrt W
    # Z = A-1 * W + B
    # Z wrt W = A-1
    # L wrt W = grad l * A-1

    # output grad -> batch x output
    # w           -> input x output
    # a-1 = input -> batch x input
    # a-1.T * output grad
    # input x batch * batch x ouput -> input x output
    weights_grad = np.dot(input.T, output_grad)


    # b = b - lr * L wrt B
    # Z wrt b = 1
    # L wrt b = grad l
    bias_grad = output_grad

    updated_weights = weights - learning_rate * weights_grad
    updated_bias = bias - learning_rate * bias_grad
    return updated_weights, updated_bias
