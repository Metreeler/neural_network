import numpy as np
import cv2 as cv
import os
import shutil
from flat_layer import FlatLayer
from flatten_layer import FlattenLayer
from sigmoid_layer import SigmoidLayer
from softmax_layer import SoftmaxLayer
from relu_layer import ReluLayer
from neural_network import NeuralNetwork
from convolution_layer import ConvolutionLayer

if __name__ == "__main__":
    
    print("Loading train values ...")
    
    arr = np.loadtxt("mnist_data/mnist_train.csv",
                    delimiter=",", dtype=int, skiprows=1)
    x_train = arr[:, 1:] / 255
    
    max_value = np.unique(arr[:, 0]).size
    y_train = np.eye(max_value)[arr[:, 0]]
    
    x_train = np.reshape(x_train, (60000, 1, 28, 28))
    # cv.imwrite("data/mnist_picture/0.png", x_train[0])
    # cv.imwrite("data/mnist_picture/1.png", x_train[1])
    # cv.imwrite("data/mnist_picture/2.png", x_train[2])
    
    # conv1 = ConvolutionLayer(3, 1, 3, 1)
    
    # tmp = conv1.forward_propagation(x_train[:3])
    
    # for i in range(tmp.shape[0]):
    #     cv.imwrite("data/" + str(i) + ".png", tmp[i])
    
    print("Loading test values ...")
    
    arr = np.loadtxt("mnist_data/mnist_test.csv",
                    delimiter=",", dtype=int, skiprows=1)
    x_test = arr[:, 1:] / 255
    
    max_value = np.unique(arr[:, 0]).size
    y_test = np.eye(max_value)[arr[:, 0]]
    
    print("Initiating neural network ...")
    
    layers = [FlattenLayer(),
              FlatLayer(784, 32),
              ReluLayer(),
              FlatLayer(32, 32),
              SigmoidLayer(),
              FlatLayer(32, 10),
              SoftmaxLayer()]
    
    print(np.max(x_train))
    nn = NeuralNetwork(layers, 
                       0.001, 
                       x_train, 
                       y_train, 
                       x_test,
                       y_test,
                       0.2,
                       128)
    
    print("Training ...")
    
    nn.train(20)
    
    print("Testing ...")
    
    output = nn.test()
    
    folders = range(10)
    
    for folder in folders:
        folder_path = "data/" + str(folder)
        
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    result = np.argmax(output, axis=1)
    expected_result = np.argmax(y_test, axis=1)
    
    # for i in range(output.shape[0]):
    #     if result[i] != expected_result[i]:
    #         file_path = "data/" + str(expected_result[i]) + "/" + str(result[i]) + "_" + str(i) + ".png"
    #         cv.imwrite(file_path, np.reshape(x_test[i], (28, 28, 1)) * 255)
    