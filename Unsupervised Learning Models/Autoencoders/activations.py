import numpy as np

class activations:

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_deriv(x):
        s = activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_deriv(x):
        return 1 - np.tanh(x)**2