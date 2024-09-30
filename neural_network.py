import numpy as np
from layer import Layer


def loss_function(array, y):
    return (array - y) ** 2

def loss_function_derivative(array, y):
    return array - y

def accuracy(array, y):
    result = np.argmax(array, axis=1)
    expected_result = np.argmax(y, axis=1)
    acc = np.where(result == expected_result, 1, 0)
    
    return np.sum(acc) / acc.shape[0]

def loss_accuracy(array, y):
    result = np.argmax(array, axis=1)
    expected_result = np.argmax(y, axis=1)
    acc = np.where(result == expected_result, 1, 0)
    
    return np.mean((array - y) ** 2), np.sum(acc) / acc.shape[0]

class NeuralNetwork:
    def __init__(self, 
                 layers: list[Layer], 
                 learning_rate,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 validation_size,
                 batch_size) -> None:
        
        self.layers = layers
        self.learning_rate = learning_rate
        
        self.validation_size = validation_size
        
        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)
        validation_length = int(x_train.shape[0] * self.validation_size)
        
        self.x_train = x_train[idx[validation_length:]]
        self.y_train = y_train[idx[validation_length:]]
        
        self.x_validation_set = x_train[idx[:validation_length]]
        self.y_validation_set = y_train[idx[:validation_length]]
        
        self.x_test = x_test
        self.y_test = y_test
        
        self.batch_size = batch_size
        self.batches = self.x_train.shape[0] // self.batch_size
    
    def train(self, epoch):
        
        for ep in range(epoch):
            print("Epoch :", ep, end=" ")
            
            idx = np.arange(self.x_train.shape[0])
            np.random.shuffle(idx)
            
            for i in range(self.batches):
                x = self.x_train[idx[(i * self.batch_size):((i + 1) * self.batch_size)]]
                y = self.y_train[idx[(i * self.batch_size):((i + 1) * self.batch_size)]]
                
                out = self.forward(x)
                
                self.backward(out, y)
            
            loss, acc = loss_accuracy(out, y)
            # print("Train set      : ", "Loss :", np.round(loss, decimals=3),"Accuracy :",  np.round(acc, decimals=3))
            
            validation = self.forward(self.x_validation_set)
            loss, acc = loss_accuracy(validation, self.y_validation_set)
            print("Validation set : ", "Loss :", np.round(loss, decimals=3),"Accuracy :",  np.round(acc, decimals=3))
    
    def test(self):
        x = self.x_test
        
        for layer in self.layers:
            x = layer.forward_propagation(x)

        print(accuracy(x, self.y_test))
        return x

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward_propagation(x)
        return x
    
    def backward(self, forward_result, y):
        x = loss_function_derivative(forward_result, y)
        for layer in reversed(self.layers):
            x = layer.back_propagation(x, self.learning_rate)
