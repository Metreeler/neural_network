import numpy as np


class Layer:
    def __init__(self) -> None:
        self.input_vector = None
    
    def forward_propagation(self, input_vector):
        return np.array([])
    
    def back_propagation(self, partial_derivative):
        return np.array([])