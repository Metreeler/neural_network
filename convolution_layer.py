import numpy as np
import cv2 as cv
from layer import Layer

import json

class ConvolutionLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernels = np.random.rand(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size) - 0.5
        self.stride = stride
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        _, row, col = input_vector.shape
        _, _, row_k, col_k = self.kernels.shape
        row_out = int((row - row_k) / self.stride) + 1
        col_out = int((col - col_k) / self.stride) + 1
        
        out = np.zeros((self.out_channels, row_out, col_out))
        
        for n in range(self.out_channels):
            for m in range(self.in_channels):
                for i in range(row_k):
                    for j in range(col_k):
                        out[n] += self.kernels[m, n, i, j] * input_vector[m, i:(i + row_out * self.stride):self.stride, j:(j + col_out * self.stride):self.stride]
        
        out /= np.max(np.abs(out))
        return out
    
    def back_propagation(self, partial_derivative, learning_rate):
        old_kernels = self.kernels
        
        _, _, row_k, col_k = self.kernels.shape
        
        _, row_p, col_p = partial_derivative.shape
        
        for n in range(self.out_channels):
            for m in range(self.in_channels):
                for i in range(row_k):
                    for j in range(col_k):
                        self.kernels[m, n, i, j] -= np.sum(self.input_vector[m, i:(i + row_p * self.stride):self.stride, j:(j + col_p * self.stride):self.stride] * partial_derivative[n]) * learning_rate
                        
                self.kernels[m, n] /= np.max(np.abs(self.kernels[m, n]))
        
        out = np.zeros(self.input_vector.shape)
        
        for m in range(self.in_channels):
            for n in range(self.out_channels):
                for i in range(row_k):
                    for j in range(col_k):
                        out[m, i:(i + row_p * self.stride):self.stride, j:(j + col_p * self.stride):self.stride] += partial_derivative[n] * old_kernels[m, n, i, j]
                        
        return out
    
    def render_kernels(self, filename, space=3):
        out = np.ones((space + self.in_channels * (space + self.kernel_size), space + self.out_channels * (space + self.kernel_size), 3))
        
        for m in range(self.in_channels):
            for n in range(self.out_channels):
                out[(space + m * (self.kernel_size + space)):((m + 1) * (space + (self.kernel_size))), 
                    (space + n * (self.kernel_size + space)):((n + 1) * (space + (self.kernel_size)))] = visualize_output(self.kernels[m, n])
        
        cv.imwrite(filename, out * 255)


def loss_function(array, y):
    return (array - y) ** 2

def visualize_output(src):
    
    src_g = np.where(src > 0, src, 0)
    src_r = np.where(src < 0, -src, 0)
    
    row, col = src.shape
    src_3d = np.zeros((row, col, 3))
    
    src_3d[:, :, 1] = src_g
    src_3d[:, :, 2] = src_r
    
    return src_3d

def normalize(src):
    mean = np.mean(src)
    std = np.std(src)
    return (src - mean) / std

if __name__ == "__main__":
    
    # img = np.zeros((32, 32))
    # img[3:-3, 3:12] = 1
    # img[3:12, -12:-3] = 1
    # img = cv.circle(img, (23, 23), 5, (1, 1, 1), -1)
    # cv.imwrite("data/img_test.png", img * 255)
    
    # # input_layer = cv.imread("data/planet.jpg")
    input_layer = cv.imread("data/img_test.png")[:, :, :] / 255
    print(input_layer.shape)
    
    input_layer = np.transpose(input_layer, (2, 0, 1))
    
    with open("json/desired_output.json") as f:
        data = json.load(f)
    desired_output = np.array(data["outputs"])[:, :, :]
    
    conv1 = ConvolutionLayer(3, 1, 3, 1)
    conv2 = ConvolutionLayer(1, 1, 3, 1)
    
    # conv1.kernels[:, 0] = [[-1, -1, -1],
    #                        [-1, 8, -1],
    #                        [-1, -1, -1]]
    # conv1.kernels[:, 0] /= 8
    
    # conv1.kernels[:, 1] = [[-1, 0, 1],
    #                        [-2, 0, 2],
    #                        [-1, 0, 1]]
    # conv1.kernels[:, 1] /= 4
    
    # conv1.kernels[:, 2] = [[-1, -2, -1],
    #                        [0, 0, 0],
    #                        [1, 2, 1]]
    # conv1.kernels[:, 2] /= 4
    
    # conv1.kernels[:, 0] = [[-1, 0, 1],
    #                        [-2, 0, 2],
    #                        [-1, 0, 1]]
    # conv1.kernels[:, 0] /= 4
    
    # conv2.kernels[:, 0] = [[-1, -2, -1],
    #                        [0, 0, 0],
    #                        [1, 2, 1]]
    # conv2.kernels[:, 0] /= 4
    
    # result = conv1.forward_propagation(input_layer)
    # result = conv2.forward_propagation(result)
    
    # for i in range(result.shape[0]):
    #     print("#" * 10, i, "#" * 10)
    #     # print(np.min(result[i]), np.max(result[i]))
    #     print(result.shape)
    #     cv.imwrite("data/test_two_layers_result_" + str(i) + ".png", visualize_output(result[i]) * 255)
    
    # data = {
    #     "outputs": result.tolist()
    # }
    
    # with open("json/desired_output.json", "w") as f:
    #     json.dump(data, f)
    
    # loss = result - desired_output
    
    # print(loss.shape)
    
    # for i in range(result.shape[0]):
    #     # print("#" * 10, i, "#" * 10)
    #     # print(np.min(result[i]), np.max(result[i]))
        
    #     cv.imwrite("data/out_test_before_result_" + str(i) + ".png", visualize_output(result[i]) * 255)
    
    
    
    
    for i in range(1000):
        # print("#" * 10, i, "#" * 10)
        
        result = conv1.forward_propagation(input_layer)
        result = conv2.forward_propagation(result)
        
        loss = result - desired_output
        
        # print("loss  :", np.min(loss), np.max(loss))
        # print("result:", np.min(result), np.max(result))
        # print("diff  :", np.min(loss) - np.min(result), np.max(loss) - np.max(result))
        # print(conv1.kernels[:, 0])
        # print(conv1.kernels[:, 1])
        # print(conv1.kernels[:, 2])
        
        out = conv2.back_propagation(loss, 0.001)
        
        conv1.back_propagation(out, 0.001)
    
    for i in range(result.shape[0]):
        
        cv.imwrite("data/out_test_result_" + str(i) + ".png", visualize_output(result[i]) * 255)
        
    # data = {
    #     "outputs": conv1.kernels.tolist()
    # }
    
    # with open("json/kernels.json", "w") as f:
    #     json.dump(data, f)
        
    conv1.render_kernels("data/final_kernels_1.png")
    conv2.render_kernels("data/final_kernels_2.png")
    
    