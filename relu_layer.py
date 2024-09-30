import numpy as np
from layer import Layer

def relu(array):
    return 

class ReluLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        return np.where(self.input_vector > 0, self.input_vector, 0)

    def back_propagation(self, partial_derivative, _):
        return np.where(self.input_vector > 0, 1, 0) * partial_derivative
    