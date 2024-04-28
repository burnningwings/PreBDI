from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from .TimeDistributed import TimeDistributed
from .CNNLayer import CNNLayer
from .mlp import MultiLayerPerceptron
from .CrossFormer import Crossformer
from einops import rearrange, repeat
from .Modules.GRL import GradientReverseLayer

def baseline_factory(config, data):
    task = config['task']
    # 这里修改序列长度为 L * N
    # feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    feat_dim = 3

    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x
    if (task == "classification"):
        if config['model'] == 'LSTM':
            return LSTM(config['num_nodes'], out_len=config['output_len'], if_domain=False)
        elif config['model'] == '1DCNN':
            return CNN1D(config['num_nodes'], out_len=config['output_len'], if_domain=False)
        elif config['model'] == 'HCG':
            return HCG(feature_dim=1, out_len=config['output_len'], if_domain=False)
        elif config['model'] == 'DANN':
            return DANN(input_shape=(config['input_len'], config['num_nodes']), num_class=config['output_len'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))

class LSTM(nn.Module):
    """
    LSTM implements
    """
    def __init__(self, input_size = 20, hidden_size = 128, num_layers = 2, out_len = 50, if_domain = True):
        super(LSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) # (B, L, D)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=512)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=512, out_features= out_len)

        # 分类任务
        # self.out = nn.Softmax(dim=1)

        self.num_classes = out_len
        self.if_domain = if_domain
        self.hidden_size = hidden_size
        if if_domain:
            self.build_discriminator()

    def build_discriminator(self):
        self.discriminator = nn.Sequential()
        self.discriminator.append(GradientReverseLayer())
        self.discriminator.append(nn.Linear(in_features=self.hidden_size, out_features=128))
        self.discriminator.append(nn.BatchNorm1d(128))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=128, out_features=64))
        self.discriminator.append(nn.BatchNorm1d(64))
        # self.discriminator.append(nn.Dropout(0.5))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=64, out_features=2))
        self.discriminator.append(nn.BatchNorm1d(2))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Softmax(dim=1))

    def forward(self, X, mask):
        input = X[:, :, :, 0]
        hidden = self.lstm1(input)[0][:,-1,:]
        predict = self.fc2(self.act(self.fc1(hidden)))

        # 分类任务
        # predict = self.out(predict)

        if self.if_domain:
            domain = self.discriminator(hidden)
            return predict, domain
        return predict

class CNN1D(nn.Module):
    """
    1DCNN implements
    """
    def __init__(self, input_dim = 20, out_len = 50, if_domain = True):
        super(CNN1D, self).__init__()

        self.Con1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=16, padding='same')
        self.act1 = nn.LeakyReLU()
        self.Con2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16, padding='same')
        self.BN1 = nn.BatchNorm1d(32)
        self.act2 = nn.LeakyReLU()
        self.Pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.Con3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, padding='same')
        self.act3 = nn.LeakyReLU()
        self.Con4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, padding='same')
        self.BN2 = nn.BatchNorm1d(64)
        self.act4 = nn.LeakyReLU()
        self.Pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.Con5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, padding='same')
        self.act5 = nn.LeakyReLU()
        self.Con6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, padding='same')
        self.BN3 = nn.BatchNorm1d(128)
        self.act6 = nn.LeakyReLU()
        self.Pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=512, out_features=512)
        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(in_features=1024, out_features=out_len)

        # 分类任务
        self.out = nn.Softmax(dim=1)

        self.num_classes = out_len
        self.if_domain = if_domain
        self.hidden_size = 256
        self.num_classes = out_len
        if if_domain:
            self.build_discriminator()

    def build_discriminator(self):
        self.discriminator = nn.Sequential()
        self.discriminator.append(GradientReverseLayer())
        self.discriminator.append(nn.Linear(in_features=self.hidden_size, out_features=128))
        self.discriminator.append(nn.BatchNorm1d(128))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=128, out_features=64))
        self.discriminator.append(nn.BatchNorm1d(64))
        # self.discriminator.append(nn.Dropout(0.5))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=64, out_features=2))
        self.discriminator.append(nn.BatchNorm1d(2))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Softmax(dim=1))

    def forward(self, X, mask):
        input = X[:, :, :, 0]
        input = rearrange(input, 'b l n-> b n l')
        hidden = self.Pool1(self.act2(self.BN1(self.Con2(self.act1(self.Con1(input))))))
        hidden = self.Pool2(self.act4(self.BN2(self.Con4(self.act3(self.Con3(hidden))))))
        hidden = self.Pool3(self.act6(self.BN3(self.Con6(self.act5(self.Con5(hidden))))))
        hidden = self.flatten(hidden)
        predict = self.fc1(self.act(self.fc(hidden)))

        # 分类任务
        predict = self.out(self.fc1(predict))

        if self.if_domain:
            domain = self.discriminator(hidden)
            return predict,domain
        return predict

class HCG(nn.Module):
    """
    HCG implements

    """
    def __init__(self, feature_dim = 3, conv_channels = 32, hidden_size = 128, out_len = 200, if_domain = True):
        super(HCG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=feature_dim, out_channels=conv_channels, kernel_size=(5, 20), padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(5, 20), padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(5, 20), padding='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv4 = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(5, 20), padding='same')
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2))

        self.gru = nn.GRU(input_size=32, hidden_size=hidden_size, num_layers=2)
        self.flatten = nn.Flatten()

        self.if_domain = if_domain
        self.hidden_size = hidden_size
        self.num_classes = out_len
        if if_domain:
            self.build_discriminator()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=512)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=out_len)

        # 分类任务
        # self.out = nn.Softmax(dim=1)

    def build_discriminator(self):
        self.discriminator = nn.Sequential()
        self.discriminator.append(GradientReverseLayer())
        self.discriminator.append(nn.Linear(in_features=self.hidden_size, out_features=128))
        self.discriminator.append(nn.BatchNorm1d(128))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=128, out_features=64))
        self.discriminator.append(nn.BatchNorm1d(64))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=64, out_features=2))
        self.discriminator.append(nn.BatchNorm1d(2))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Softmax(dim=1))


    def forward(self, X, mask):
        input = X[:, :, :, [0]]
        input = rearrange(input, 'b l n d-> b d l n')
        hidden = self.maxpool1(self.conv1(input))
        hidden = self.maxpool2(self.conv2(hidden))
        hidden = self.maxpool3(self.conv3(hidden))
        hidden = self.maxpool4(self.conv4(hidden))
        hidden = self.gru(hidden[:, :, :, 0])
        hidden = hidden[0][:,-1,:]
        # hidden = self.flatten(hidden[0])

        cla = self.fc1(hidden)
        cla = self.act(cla)
        predict = self.fc2(cla)

        # 分类任务
        # predict = self.out(predict)

        if self.if_domain:
            domain = self.discriminator(hidden)
            return predict, domain
        return predict


class DANN(nn.Module):
    def __init__(self, input_shape, num_class):
        super(DANN, self).__init__()

        _, N = input_shape
        ## Feature extractor
        self.Featurizer = nn.Sequential()
        self.Featurizer.append(nn.Conv1d(in_channels=N, out_channels=16, kernel_size=14, stride=1))
        self.Featurizer.append(nn.BatchNorm1d(16))
        self.Featurizer.append(nn.MaxPool1d(kernel_size=4, stride=2))
        self.Featurizer.append(nn.LeakyReLU())
        self.Featurizer.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=14, stride=1))
        self.Featurizer.append(nn.BatchNorm1d(32))
        self.Featurizer.append(nn.MaxPool1d(kernel_size=4, stride=2))
        self.Featurizer.append(nn.LeakyReLU())
        self.Featurizer.append(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=14, stride=1))
        self.Featurizer.append(nn.BatchNorm1d(64))
        self.Featurizer.append(nn.MaxPool1d(kernel_size=4, stride=2))
        self.Featurizer.append(nn.LeakyReLU())
        self.Featurizer.append(nn.Flatten())

        ## Label Classifier
        self.Classifier = nn.Sequential()
        self.Classifier.append(nn.Linear(in_features=64, out_features=128))
        self.Classifier.append(nn.BatchNorm1d(128))
        self.Classifier.append(nn.LeakyReLU())
        self.Classifier.append(nn.Linear(in_features=128, out_features=num_class))
        self.Classifier.append(nn.BatchNorm1d(num_class))
        self.Classifier.append(nn.LeakyReLU())
        self.Classifier.append(nn.Softmax(dim=1))

        ## Domain Discriminator
        self.discriminator = nn.Sequential()
        self.discriminator.append(GradientReverseLayer())
        self.discriminator.append(nn.Linear(in_features=64, out_features=128))
        self.discriminator.append(nn.BatchNorm1d(128))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=128, out_features=64))
        self.discriminator.append(nn.BatchNorm1d(64))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Linear(in_features=64, out_features=2))
        self.discriminator.append(nn.BatchNorm1d(2))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(nn.Softmax(dim=1))


    def forward(self, x):

        input = x[..., 0]
        input = input.permute(0, 2, 1).contiguous()

        hidden = self.Featurizer(input)

        label_out = self.Classifier(hidden)
        domain_out = self.discriminator(hidden)
        return label_out, domain_out
