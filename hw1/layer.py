import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.cache = np.zeros((1,in_features))

    def forward(self, input):
        output = input.dot(self.weight) + self.bias
        self.cache = input
        return output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.cache.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis = 1, keepdims = True)
        
        return input_grad

## by yourself .Finish your own NN framework
class ACTIVITY1(_Layer):
    def __init__(self):
        pass
    def forward(self, input):
        output = np.maximum(0.01*input,input)
        self.cache = input
        return output

    def backward(self, output_grad):
        input_grad = np.array(output_grad , copy = True)
        input_grad[self.cache <= 0] *= 0.01 
        return input_grad

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass
    def forward(self, input, target):

        '''Softmax'''
        predict = np.exp(input - np.max(input))
        predict = predict / predict.sum(keepdims = True) 
        '''Average loss'''
        your_loss = -np.mean(target * np.log(predict + 1e-8))
        self.cache1 = predict
        self.cache2 = target
        return predict, your_loss

    def backward(self):
        input_grad = self.cache1 - self.cache2
        

        return input_grad
    
    