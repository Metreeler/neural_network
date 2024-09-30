import numpy as np
from layer import Layer


class NormalizationLayer(Layer):
    def __init__(self, in_channels, layer_size) -> None:
        super().__init__()
        
    
    def forward_propagation(self, input_vector):
        return np.array([])
    
    def back_propagation(self, partial_derivative, learning_rate):
        return np.array([])
