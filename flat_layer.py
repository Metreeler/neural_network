import numpy as np
from layer import Layer


class FlatLayer(Layer):
    def __init__(self, in_channels, layer_size) -> None:
        super().__init__()
        self.weights = np.random.rand(in_channels, layer_size) * 2 - 1
        self.biases = np.random.rand(1, layer_size)
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        out = np.matmul(input_vector, self.weights) + self.biases
        return out
    
    def back_propagation(self, partial_derivative, learning_rate):
        old_weights = self.weights
        self.weights -= np.matmul(self.input_vector.T, partial_derivative) * learning_rate
        self.biases -= np.mean(partial_derivative, axis=0) * learning_rate
        return np.dot(partial_derivative, old_weights.T)
