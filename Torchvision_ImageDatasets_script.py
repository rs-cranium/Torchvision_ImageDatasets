# Please use it when you install it.
# !pip install torch==0.4.1
# !pip install torchvision==0.2.1
# !pip install matplotlib==2.1.2
import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
# =========================================
# MNIST
# Dataset of handwritten numbers from 0 to 9.
# =========================================
mnist_data = datasets.MNIST(root="mnist_root", download=True)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(mnist_data[i][0].convert('RGB'))

plt.show()
# =========================================
# Fashion-MNIST
# Dataset of images of clothes, shoes and bags.
# =========================================
fashion_data = datasets.FashionMNIST(root="torchData/fashion_root", download=True)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(fashion_data[i][0].convert('RGB'))
    
plt.show()
# =========================================
# EMNIST
# Data set consisting of handwritten numbers, 
# uppercase and lowercase alphabetic images.
# =========================================
emnist_data = datasets.EMNIST(root="emnist_root", split="balanced", train=True, download=True)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(emnist_data[i][0].convert('RGB'))
    
plt.show()
# =========================================
# SVHN(The Street View House Numbers)
# Dataset of photos of residence numbers retrieved from 
# Google Street View.
# =========================================
svhn_data = datasets.SVHN(root="svhn_root", download=True)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(svhn_data[i][0].convert('RGB'))
    
plt.show()
# =========================================
# CIFAR10
# Data set of color photographs of animals and airplanes.
# =========================================
cifar10_data = datasets.CIFAR10(root="cifar10_root", download=True)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(cifar10_data[i][0].convert('RGB'))
    
plt.show()
# =========================================
# CIFAR100
# This dataset is the same type as CIFAR10, 
# but it has 100 classes.
# =========================================
cifar100_data = datasets.CIFAR100(root="cifar100_root", download=True)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(cifar100_data[i][0].convert('RGB'))
    
plt.show()
# =========================================
# STL10
# It is the same type of data set as CIFAR10, 
# but with higher resolution.
# =========================================
stl_data = datasets.STL10(root="stl_root", download=True)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(stl_data[i][0].convert('RGB'))
    
plt.show()

