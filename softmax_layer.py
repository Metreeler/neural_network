import numpy as np
from layer import Layer

class SoftmaxLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        e_x = np.exp(input_vector)
        out = e_x / np.sum(e_x, axis=1)[:, np.newaxis]
        # print("here")
        # print(np.sum(e_x, axis=1)[:, np.newaxis].shape)
        # print(out.shape)
        return out

    def back_propagation(self, partial_derivative, _):
        return partial_derivative
