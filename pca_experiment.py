import pdb
from pdb import set_trace as bp
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocess
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn import covariance as cov
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pyRMT as rmt
import seaborn as sns

from skorch import NeuralNetClassifier

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data(data_dir='/media/l7/data_storage1/datasets', process=True):
    data_dir = data_dir

    data = np.load(data_dir + '/numpy_fashion_mnist_data.npy', allow_pickle=True)
    target = np.load(data_dir + '/numpy_fashion_mnist_target.npy', allow_pickle=True)
    target = np.array(target, dtype=int)
    if process:
        processed_data = data / 255
        standard_scaler = preprocess.StandardScaler()
        standard_scaler.fit(processed_data)
        processed_data = standard_scaler.transform(processed_data)
        return processed_data, target
    else:
        return data, target


def plot_numpy_images(images,targets, img_shape=[28,28], n_rows=1):
    plt.figure(figsize=(10,10))
    for index, img, label in zip(np.arange(images.shape[0]), images, targets):
        plt.subplot(n_rows, images.shape[0] // n_rows, index + 1)
        img = img.reshape(img_shape)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.Greys)
#         bp()
        plt.xlabel(classes[label])
    plt.tight_layout(w_pad=0)


def plot_confusion_matrix(y_pred, y, prefix_information='',
                          dataset_name='', save_results=False,
                          y_pred_is_predicted_classes=False):
    """
    PLotting confusion matrix of different datasets (train, test, and validation).
    :param y_pred:
    :param y:
    :param prefix_information:
    :param dataset_name:
    :param save_results:
    :param y_pred_is_predicted_classes:
    :return:
    """
    if 'Tensor' in str(type(y)):
        y = y.numpy()
    c_matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    result_accuracy_string = "Accuracy of Net: {:.2f}".format(accuracy)
    print(result_accuracy_string)
    print("\nClassification report:")
    classfication_report_string = classification_report(y, y_pred)
    print(classfication_report_string)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_title(" Confusion matrix")
    sns.heatmap(c_matrix, cmap='Blues',
                annot=True, xticklabels=classes,
                yticklabels=classes, fmt='g', cbar=False)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xlabel('predictions')
    ax.set_ylabel('true labels')
    plt.tight_layout(pad=1.08, w_pad=10, h_pad=10)
    # plt.show()
    return accuracy


def compute_pca(data, number_of_eigen_vectors=None,
                method_name='lwe'):
    if method_name == 'lwe':
        data_cov = cov.ledoit_wolf(data, assume_centered=True)[0]
    elif method_name == 'sce':
        data_cov = np.cov(data, rowvar=False)
    elif method_name == 'rmt':
        data_cov = rmt.optimalShrinkage(data, return_covariance=True)
    else:
        raise Exception('pca method_name incorrect')
    eigen_values, eigen_vectors = np.linalg.eig(data_cov)
    if number_of_eigen_vectors is None:
        decorrelated_data = data.dot(eigen_vectors)
    else:
        decorrelated_data = data.dot(eigen_vectors[:, :number_of_eigen_vectors])
    return decorrelated_data


def setup_data_sets(data, target, test_size, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=test_size, random_state=random_state)
    x_train = torch.tensor(x_train).float().unsqueeze(1)
    y_train = torch.tensor(y_train).long()
    x_test = torch.tensor(x_test).float().unsqueeze(1)
    y_test = torch.tensor(y_test).long()
    return x_train, x_test, y_train, y_test


class LeNet5(nn.Module):
    def __init__(self, n_channels, n_features):
        super(LeNet5, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 16, 5)
        self.name = 'LeNet5'
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d_2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #         bp()
        x = self.max_pool(self.relu(self.conv2d_1(x)))  # [50, 16, 1, 1]
        # bp()
        x = self.max_pool(self.relu(self.conv2d_2(x)))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(n_features, n_channels, x_train, y_train,
                device=torch.device('cuda'),
                model=None, n_epochs=10):
    if model is None:
        model = NeuralNetClassifier(module=LeNet5,
                                    criterion=nn.CrossEntropyLoss,
                                    module__n_channels= n_channels,
                                    module__n_features=n_features,
                                    optimizer=optim.Adam,
                                    optimizer__lr=0.001,
                                    max_epochs=n_epochs,
                                    batch_size=32,
                                    iterator_train__shuffle=True,
                                    device=device,
                                    warm_start=True,
                                   )
    model.fit(x_train, y_train, epochs=n_epochs)
    pred_train = model.predict(x_train)
    accuracy = plot_confusion_matrix(pred_train, y_train, )
    return model, pred_train, accuracy


def test_model(model, test_data, test_targets):
    pred_test = model.predict(test_data)
    accuracy = plot_confusion_matrix(pred_test, test_targets)
    return pred_test, accuracy

