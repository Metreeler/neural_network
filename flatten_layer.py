import numpy as np
import cv2 as cv
from layer import Layer


class FlattenLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels_shape = None
    
    def forward_propagation(self, input_vector):
        self.in_channels_shape = input_vector.shape
        return np.reshape(input_vector, (self.in_channels_shape[0], np.prod(self.in_channels_shape[1:])))
    
    def back_propagation(self, partial_derivative, _):
        return np.reshape(partial_derivative, self.in_channels_shape)


    
    
    