import cv2
import numpy as np
from torch import nn
from scipy import *


# # KL散度
#
# def asymmetricKL(P, Q):
#     return sum(P * log(P / Q))  # calculate the kl divergence between P and Q
#
#
# def symmetricalKL(P, Q):
#     return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00


# L2 损失函数


class L2loss(nn.Module):
    def __init__(self, use_gpu=False, gpu_device=None):
        self.use_gpu = use_gpu
        self.gpu_device = None
        # if use_gpu and gpu_device is not None:
        #     self.gpu_device = gpu_device

    def forward(self, target, output):
        h1 = np.zeros((224 * 224))
        h2 = np.zeros((224 * 224))
        loss = 0
        for i in range(3):
            # print("*****************************")
            # print("target.type:", type(target))
            # print("output.type:", type(output))
            # print("*****************************")
            # print("target.shape:", target.shape)
            # print("output.shape:", output.shape)
            # print("*****************************")
            # print("target[i].shape:", target[i].shape)
            # print("output[i].shape:", output[i].shape)
            target[i, :, 0] = np.cumsum(target[i])
            output[i, :, 0] = np.cumsum(output[i])

            for j in range(len(target[i])):
                if j == 0:
                    h1[:int(target[i, j, 0])] = 0
                    h2[:int(output[i, j, 0])] = 0
                elif j == len(target[i]) - 1:
                    h1[int(target[i, j, 0]):] = j
                    h2[int(output[i, j, 0]):] = j
                else:
                    h1[int(target[i, j - 1, 0]):int(target[i, j, 0])] = j
                    h2[int(output[i, j - 1, 0]):int(output[i, j, 0])] = j

            loss += np.sqrt(np.sum(np.square(np.absolute(h1 - h2))))
        return loss


if __name__ == '__main__':
    img1 = cv2.imread(r"D:\TEST_IMAGE\3.jpg")
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    hist3 = cv2.calcHist([img1], [2], None, [256], [0, 256])
    pred = np.concatenate((hist1, hist2, hist3))
    pred = pred.reshape((3, 256, 1))

    img2 = cv2.imread(r"D:\TEST_IMAGE\3.jpg")
    hist2_1 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist2_2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
    hist2_3 = cv2.calcHist([img2], [2], None, [256], [0, 256])
    target = np.concatenate((hist2_1, hist2_2, hist2_3))
    target = target.reshape((3, 256, 1))
    L2loss(pred, target)
