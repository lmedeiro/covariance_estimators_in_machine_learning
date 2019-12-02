import pdb
from pdb import set_trace as bp
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import localtime, strftime
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


data_dir = '/media/l7/data_storage1/datasets'


def load_data_from_sklearn():
    fashion_mnist = fetch_openml(name='Fashion-MNIST', cache=True, data_home=data_dir)
    np.save(data_dir + '/numpy_fashion_mnist_data.npy', fashion_mnist.data)
    np.save(data_dir + '/numpy_fashion_mnist_target.npy', fashion_mnist.target,
            allow_pickle=True)


def plot_images(images, n_rows=1):
    fig, axs = plt.subplots(n_rows, images.size(0) // n_rows)
    for ax, img in zip(axs.flat, images):
        ax.matshow(img[0].cpu().numpy(), cmap=plt.cm.Greys)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tight_layout(w_pad=0)
    

def work_with_pytorch():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        transforms.Normalize((0.0,), (1,))  # Normalizing values to Mu = 0, Sigma = 1
    ])
    data_dir = '/media/l7/data_storage1/datasets'
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    plot_images(images[:8], n_rows=2)