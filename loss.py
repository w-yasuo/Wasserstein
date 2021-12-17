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
def L2loss():
    return nn.MSELoss()

