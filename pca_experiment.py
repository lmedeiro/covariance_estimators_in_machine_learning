import pdb
from pdb import set_trace as bp
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocess
from sklearn import covariance as cov
from sklearn.metrics import confusion_matrix, \
    classification_report, accuracy_score, f1_score, precision_score, recall_score
import pyRMT as rmt
import seaborn as sns
import argparse
import os
from skorch import NeuralNetClassifier

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
LOCAL_PC = 'l7'
if os.getlogin() == LOCAL_PC:
    DATA_DIR = '/media/l7/data_storage1/datasets'
else:
    DATA_DIR = '/u/54/medeirl1/unix/dev/datasets'
MAIN_PATH = DATA_DIR + '/afib_dataset/'

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
        plt.xlabel(CLASSES[label])
    plt.tight_layout(w_pad=0)


def get_classification_metrics(y_pred, y):
    accuracy = accuracy_score(y, y_pred)
    f1_score_value = f1_score(y, y_pred, average='macro')
    precision_score_value = precision_score(y, y_pred, average='macro')
    recall_score_value = recall_score(y, y_pred, average='macro')
    classification_results = {'accuracy_score_value': accuracy, 'f1_score_value': f1_score_value,
            'precision_score_value': precision_score_value, 'recall_score_value': recall_score_value}
    return classification_results


def plot_confusion_matrix(y_pred, y, get_all_metrics=False):
    """
    PLotting confusion matrix of different datasets (train, test, and validation).
    :param y_pred: predicted y values
    :param y: true y values
    :param get_all_metrics: bool indicating whether or not to get all metrics computed
    :return: accuracy, or list with all metrics
    """
    if 'Tensor' in str(type(y)):
        y = y.numpy()
    c_matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    f1_score_value = f1_score(y, y_pred)
    precision_score_value = precision_score(y, y_pred)
    recall_score_value = recall_score(y, y_pred)
    result_accuracy_string = "Accuracy of Net: {:.2f}".format(accuracy)
    print(result_accuracy_string)
    print("\nClassification report:")
    classfication_report_string = classification_report(y, y_pred)
    print(classfication_report_string)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_title(" Confusion matrix")
    sns.heatmap(c_matrix, cmap='Blues',
                annot=True, xticklabels=CLASSES,
                yticklabels=CLASSES, fmt='g', cbar=False)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xlabel('predictions')
    ax.set_ylabel('true labels')
    plt.tight_layout(pad=1.08, w_pad=10, h_pad=10)
    # plt.show()
    if get_all_metrics:
        return {'accuracy_score_value': accuracy, 'f1_score_value': f1_score_value,
                'precision_score_value': precision_score_value, 'recall_score_value': recall_score_value}
    else:
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
        # bp()
        x = self.max_pool(self.relu(self.conv2d_1(x)))  # [50, 16, 1, 1]
        # bp()
        x = self.max_pool(self.relu(self.conv2d_2(x)))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5OneD(nn.Module):
    def __init__(self, n_channels, n_features, mid_layer_channels=10,
                 C_k_p_s_1=[5, 0, 1], M_k_s_1=[2, 2], C_k_p_s_2=[5, 0, 1],
                 M_k_s_2=[2, 2], p=[0.0, 0.0, 0.0, 0.0, 0.0]):
        """
        Replica of a LeNet5 model, with adaptable parameters
        :param n_channels: number of channels in the input data
        :param n_features: number of features in the input data
        :param mid_layer_channels: number of layers which will be produced by the
        the middle CNN layer
        :param C_k_p_s_1: list containing kernel_size, padding, and stride
        :param C_k_p_s_2: list containing kernel_size, padding, and stride
        :param M_k_s_1: list containing max pooling kernel_size and stride
        :param M_k_s_2: list containing max pooling kernel_size and stride
        :param p: dropout layer probabilities
        """
        super(LeNet5OneD, self).__init__()
        self.name = 'LetNet5OneD'
        self.conv1d_1 = nn.Conv1d(n_channels, mid_layer_channels,
                                  kernel_size=C_k_p_s_1[0], padding=C_k_p_s_1[1],
                                  stride=C_k_p_s_1[2])
        self.relu = nn.ReLU()
        self.max_pool_1 = nn.MaxPool1d(kernel_size=M_k_s_1[0],
                                       stride=M_k_s_1[1])
        current_input_size = \
            (1 + (n_features - C_k_p_s_1[0] + 2 * C_k_p_s_1[1]) / C_k_p_s_1[2]) / M_k_s_1[0]
        current_input_size = int(current_input_size)
        self.drop_out_1 = nn.Dropout(p=p[0])
        #         bp()
        self.conv1d_2 = nn.Conv1d(mid_layer_channels,
                                  2 * mid_layer_channels,
                                  kernel_size=C_k_p_s_2[0],
                                  padding=C_k_p_s_2[1],
                                  stride=C_k_p_s_2[2])
        current_input_size = \
            (1 + (current_input_size - C_k_p_s_2[0] + 2 * C_k_p_s_2[1]) / C_k_p_s_2[2]) / M_k_s_1[0]
        current_input_size = int(current_input_size)
        self.drop_out_2 = nn.Dropout(p=p[1])
        #         bp()
        self.max_pool_2 = nn.MaxPool1d(kernel_size=M_k_s_2[0],
                                       stride=M_k_s_2[1])
        #         bp()
        self.fc1 = nn.Linear(2 * mid_layer_channels * current_input_size, 120)
        self.drop_out_3 = nn.Dropout(p=p[2])
        self.fc2 = nn.Linear(120, 84)
        self.drop_out_4 = nn.Dropout(p=p[3])
        self.fc3 = nn.Linear(84, 10)
        self.drop_out_5 = nn.Dropout(p=p[4])
        self.soft_max = nn.Softmax(dim=1)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        #         bp()
        x = x.float()
        x = self.max_pool_1(self.relu(self.drop_out_1(self.conv1d_1(x))))
        # bp()
        x = self.max_pool_2(self.relu(self.drop_out_2(self.conv1d_2(x))))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.relu(self.drop_out_3(self.fc1(x)))
        x = self.relu(self.drop_out_4(self.fc2(x)))
        x = self.relu(self.drop_out_5(self.fc3(x)))
        x = self.fc4(x)
        x = self.soft_max(x)
        return x


def train_model(x_train, y_train,
                device=torch.device('cuda'), pca_data=False,
                model=None, n_epochs=10, plot_results=False):
    if 'numpy' in str(type(x_train)):
        try:
            x_train = torch.tensor(x_train).float().unsqueeze(1)
            y_train = torch.tensor(y_train).long()
        except:
            # bp()
            x_train = torch.tensor(np.abs(x_train)).float().unsqueeze(1)
            y_train = torch.tensor(y_train).long()
            print('exception')
    if model is None:
        if pca_data:
            model = NeuralNetClassifier(module=LeNet5OneD,
                                        criterion=nn.CrossEntropyLoss,
                                        module__n_channels=x_train.shape[1],
                                        module__n_features=x_train.shape[2],
                                        optimizer=optim.Adam,
                                        optimizer__lr=0.001,
                                        max_epochs=n_epochs,
                                        batch_size=32,
                                        iterator_train__shuffle=True,
                                        device=device,
                                        warm_start=True,
                                       )
        else:
            model = NeuralNetClassifier(module=LeNet5,
                                        criterion=nn.CrossEntropyLoss,
                                        module__n_channels=None,
                                        module__n_features=None,
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
    if plot_results:
        accuracy = plot_confusion_matrix(pred_train, y_train, )
        return model, pred_train, accuracy
    else:
        classification_results = get_classification_metrics(pred_train, y_train)
        return model, pred_train, classification_results


def test_model(model, test_data, test_targets, plot_results=False, pca_data=False):
    if 'numpy' in str(type(test_data)):
        if not pca_data:
            test_data = test_data.reshape(test_data.shape[0], 28, 28)
        test_data = torch.tensor(test_data).float().unsqueeze(1)
        test_targets = torch.tensor(test_targets).long()
    pred_test = model.predict(test_data)
    if plot_results:
        accuracy = plot_confusion_matrix(pred_test, test_targets, )
        return model, pred_test, accuracy
    else:
        classification_results = get_classification_metrics(pred_test, test_targets)
        return model, pred_test, classification_results


def load_pca_data(data, target, test_sample_size, method_names,
                  load_stored=True,):
    if load_stored:
        main_path = DATA_DIR + '/afib_dataset/'
        experiment_name = 'pca_experiment'
        suffix = '.npy'
        print('Loading stored data')

        test_data = {}
        for name in method_names:
            if name == 'raw':
                pass
            else:
                test_data[name] = np.load(main_path + experiment_name + '_' + name + '_test' + suffix)
        test_target = np.load(main_path + experiment_name + '_test_target' + suffix)

    else:
        print('Computing new test and train data')
        test_target = target[:test_sample_size]
        sce_data = compute_pca(data, method_name='sce')
        test_data = {}
        test_data['sce'] = sce_data[:test_sample_size, :]
        del sce_data
        rmt_data = compute_pca(data, method_name='rmt')
        test_data['rmt'] = rmt_data[:test_sample_size, :]
        del rmt_data
        lwe_data = compute_pca(data, method_name='lwe')
        test_data['lwe'] = lwe_data[:test_sample_size, :]
        del lwe_data
    test_data['raw'] = data[:test_sample_size, :]
    data = data[test_sample_size:, :]
    target = target[test_sample_size:]
    return data, target, test_data, test_target


def pca_experiment(sample_numbers, method_names, metric_names,
                   number_of_eigen_vectors=None, test_size=0.25,
                   n_epochs=5):
    data, target = load_data(data_dir=DATA_DIR)
    print(data.shape)
    print(target.shape)
    test_sample_size = int(0.25 * data.shape[0])
    random_state = 42

    data, target, test_data, test_target = load_pca_data(data, target, test_sample_size, method_names,
                                                         load_stored=True)
    # x_train, x_test, y_train, y_test = setup_data_sets(data, target, test_size, random_state=random_state)
    sample_range = range(data.shape[0])
    result_df = pd.DataFrame(columns=['sample_number', 'method_name',
                                      'metric_name', 'number_of_eigen_vectors', 'value'])
    for sample_number in sample_numbers:
        train_indexes = np.random.choice(sample_range, size=sample_number, replace=False)
        train_data = data[train_indexes, :]
        train_target = target[train_indexes]
        for method_name in method_names:
            if number_of_eigen_vectors is None:
                dimension = 784
                if method_name == 'raw':
                    # print('implement raw method')
                    raw_train_data = train_data.reshape(train_data.shape[0], 28, 28)
                    model, pred_train, classification_results = train_model(raw_train_data, train_target,
                                                                            model=None, n_epochs=n_epochs,
                                                                            plot_results=False)
                    # model, pred_test, classification_results = test_model(model,
                    #                                                       test_data[method_name],
                    #                                                       test_target,
                    #                                                       plot_results=False)
                else:
                    pca_train_data = compute_pca(train_data,
                                                 method_name=method_name)
                    # pca_train_data = train_data.reshape(pca_train_data.shape[0], 28, 28)
                    model, pred_train, classification_results = train_model(pca_train_data, train_target,
                                                                            model=None, n_epochs=n_epochs,
                                                                            plot_results=False, pca_data=True)
                    # model, pred_test, classification_results = test_model(model,
                    #                                                       test_data[method_name],
                    #                                                       test_target,
                    #                                                       plot_results=False,
                    #                                                       pca_data=True)

                for metric in metric_names:
                    index = result_df.index.size
                    result_df.loc[index] = 0
                    metric_value = classification_results[metric]
                    result_df.loc[index] = [sample_number, method_name, metric, dimension, metric_value]
            else:
                if method_name == 'raw':
                    pass
                else:
                    for dimension in number_of_eigen_vectors:
                        pca_train_data = compute_pca(train_data, number_of_eigen_vectors=dimension,
                                                     method_name=method_name)
                        # pca_train_data = train_data.reshape(pca_train_data.shape[0], 28, 28)
                        model, pred_train, classification_results = train_model(pca_train_data, train_target,
                                                                                model=None, n_epochs=n_epochs,
                                                                                plot_results=False, pca_data=True)
                        for metric in metric_names:
                            if method_name == 'raw':
                                pass
                            else:
                                index = result_df.index.size
                                result_df.loc[index] = 0
                                metric_value = classification_results[metric]
                                result_df.loc[index] = [sample_number, method_name, metric, dimension, metric_value]
        result_df.to_pickle(MAIN_PATH + 'pca_experiment_result_df.pckl')
        print('finished sample_number: {}'.format(sample_number))

    return 0


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_names", nargs='*', default=None,
                        help="Method Names: raw, lwe, sce, rmt")
    parser.add_argument("--metric_names", nargs='*', default=None,
                        help="Metric values to be stored and plotted")
    parser.add_argument("--sample_numbers", nargs='*', default=None,
                        help="Training Sample Numbers")
    parser.add_argument("--number_of_eigen_vectors", nargs='*', default=None,
                        help="List with number of eigen vectors to be used")
    parser.add_argument("--n_epochs", type=int, help="Number of epochs", default=5)
    return parser.parse_args(args)


if __name__ == "__main__":
    parser = parse_args()
    if parser.sample_numbers is not None:
        sample_numbers = []  # args.assets
        for sample_number in parser.sample_numbers:
            sample_numbers.append(int(sample_number))
    else:
        sample_numbers = [800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500,
                          2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000,
                          8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000,
                          16000, 17000, 18000, 19000,
                          20000, 30000, 40000, 50000]
        # sample_numbers = [800, 850, 900]
    if parser.method_names is not None:
        method_names = parser.method_names
    else:
        method_names = ['raw', 'sce', 'lwe', 'rmt']
    if parser.metric_names is not None:
        metric_names = parser.metric_names
    else:
        #{ 'accuracy_score_value': accuracy, 'f1_score_value': f1_score_value,
        # 'precision_score_value': precision_score_value, 'recall_score_value': recall_score_value}
        metric_names = ['f1_score_value', 'accuracy_score_value']
    if parser.n_epochs is not None:
        n_epochs = parser.n_epochs
    else:
        n_epochs = 5
    if parser.number_of_eigen_vectors is not None:
        number_of_eigen_vectors = []  # args.assets
        for dimension in parser.number_of_eigen_vectors:
            number_of_eigen_vectors.append(int(dimension))
    else:
        number_of_eigen_vectors = [100, 200, 300, 400, 500, 600, 784]
        # number_of_eigen_vectors = [100, 200, 300]
    pca_experiment(sample_numbers, method_names, metric_names,
                   number_of_eigen_vectors=number_of_eigen_vectors,
                   n_epochs=n_epochs)
