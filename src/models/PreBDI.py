from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from .CrossFormer import Crossformer
from .CrossFormer import celk
from .Modules.ConvLSTM import ConvLSTM
from .Modules.FFAttention import Attention
from .Modules.GRL import GradientReverseLayer
from .Baseline import baseline_factory
def model_factory(config, data):
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

    if (task == "imputation") or (task == "transduction"):
        if config['model'] == 'crossformer':
            return Crossformer(config['cross_data_dim'], config['cross_in_len'], config['cross_out_len'], config['cross_seg_len'],
                               config['cross_win_size'], config['cross_factor'], config['cross_d_model'], config['cross_d_ff'],
                               config['cross_n_heads'], config['cross_e_layers'], config['cross_dropout'], config['cross_baseline'])
        if config['model'] == 'celk':
            return celk(config['cross_data_dim'], config['cross_in_len'], config['cross_out_len'], config['cross_seg_len'],
                               config['cross_win_size'], config['cross_factor'], config['cross_d_model'], config['cross_d_ff'],
                               config['cross_n_heads'], config['cross_e_layers'], config['cross_dropout'], config['cross_baseline'])

    if (task == "classification") or (task == "regression"):
        if len(data.labels_df.shape) > 1:
            num_labels = data.labels_df.shape[1]
        # num_labels = 9 if task == "classification" else data.labels_df.shape[1]  # dimensionality of labels
        if config['model'] == 'PreBDI':
            return PreBDI(feat_dim, max_seq_len, config['d_model'], config['num_heads'], num_classes=num_labels, model_args=config)
        elif config['model'] == 'CrossACL':
            return CrossACL(feat_dim, max_seq_len, config['d_model'], config['num_heads'], num_classes=num_labels, model_args=config)
        elif config['model'] == 'ACLBDI':
            return ACLBDI(num_classes=num_labels, if_domain=False)
        elif config['model'] == 'DPACLBDI':
            return PreDBDI(feat_dim, max_seq_len, config['d_model'], config['num_heads'], num_classes=10, model_args=config)
        else:
            return baseline_factory(config, data)
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))

class CrossACL(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_classes, model_args=None):
        super(CrossACL, self).__init__()

        # attributes
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # model layer
        self.encoder_layer = Crossformer(model_args['cross_data_dim'], model_args['cross_in_len'], model_args['cross_out_len'], model_args['cross_seg_len'],
                               model_args['cross_win_size'], model_args['cross_factor'], model_args['cross_d_model'], model_args['cross_d_ff'],
                               model_args['cross_n_heads'], model_args['cross_e_layers'], model_args['cross_dropout'], model_args['cross_baseline'], mode='finetune')

        self.preConv = nn.Sequential()
        self.preConv.append(nn.Conv1d(in_channels=5632, out_channels=128, kernel_size=self.num_nodes, padding='same'))
        self.preConv.append(nn.BatchNorm1d(128))
        self.preConv.append(nn.LeakyReLU())
        self.ConvLSTM = ConvLSTMBlock(channels=1, hidden_dim=[self.num_nodes], kernal_size=(2,2), nums_layer=1)
        self.ConvBN = nn.BatchNorm3d(num_features=128)
        self.Flatten = nn.Flatten(2)
        self.att = Attention([None, 128, self.num_nodes * self.num_nodes])

        self.out = SimpleOut(in_dim=self.num_nodes * self.num_nodes, out_dim=self.output_len, hidden_dim=512)

    def forward(self, X, padding_masks):

        B,L,N,_ = X.shape
        time_series = X[..., 0]
        time_series_emb = self.encoder_layer(time_series)

        hidden = time_series_emb
        hidden = self.preConv(hidden)
        hidden = self.ConvLSTM(hidden, 4, 4)
        hidden = self.ConvBN(hidden)
        hidden = self.Flatten(hidden)
        hidden = self.att(hidden)
        hidden = hidden[0]

        output = self.out(hidden.contiguous().view(B, -1))

        return output

class FFAttention(nn.Module):
    def __init__(self, batch_size=128, input_len=128, d_model=640, out_dim=1):
        super(FFAttention, self).__init__()
        # Net Config
        self.input_len = input_len
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=1)
        self.w = nn.Parameter(torch.empty(d_model, 1))
        nn.init.xavier_uniform_(self.w)
        self.bias = nn.Parameter(torch.empty(self.input_len, 1))
        nn.init.zeros_(self.bias)

    def forward(self, x_emb):

        feature_dim = x_emb.shape[-1]
        input_len = x_emb.shape[1]
        em = torch.reshape(torch.matmul(torch.reshape(x_emb, (-1, feature_dim)), self.w), (-1, input_len))
        em += self.bias
        em = torch.tanh(em)
        alpha = torch.exp(em)
        alpha /= torch.sum(alpha, 1, keepdim=True) + 1e-7
        alpha = alpha.unsqueeze(dim=-1)
        w_i = x_emb * alpha
        return torch.sum(w_i, dim=1)
        # alpha = self.softmax(x_act)
        # return torch.bmm(alpha.view(-1, self.out_dim, self.input_len), x_emb)

class ConvLSTMBlock(nn.Module):
    def __init__(self, channels, hidden_dim = [32, 64], kernal_size = (3, 3), nums_layer = 2,
                batch_first = True, bias = True, return_all_layer = False):
        super(ConvLSTMBlock, self).__init__()
        self.ConvLSTM = nn.ModuleList()
        self.in_channel = channels
        for i in range(len(hidden_dim)):
            if i > 0:
                self.in_channel = hidden_dim[i - 1]
            self.ConvLSTM.append(ConvLSTM(self.in_channel, hidden_dim[i], kernal_size, nums_layer,
                                batch_first, bias, return_all_layer, padding='same'))
        # self.block = ConvLSTM(channels, hidden_dim, kernal_size, nums_layer, batch_first, bias, return_all_layer, padding='same')
    def forward(self, X, H, W):
        batch_size, input_len, num_node = X.shape
        hidden = X.reshape(batch_size, input_len, 1, H, W)
        for layer in self.ConvLSTM:
            hidden, layer_out = layer(hidden)
            hidden = hidden[0]
        return hidden

class SimpleOut(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, task="regression"):
        super(SimpleOut, self).__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.drop = nn.Dropout(0.5)
        self.act = nn.LeakyReLU()
        if task == "regression":
            self.out = nn.Linear(hidden_dim, out_features=out_dim)
        elif task == "classification":
            self.out = nn.Sequential()
            self.out.append(nn.Linear(hidden_dim, out_features=out_dim))
            # self.out.append(nn.Softmax(dim=-1))
    def forward(self, x):
        return self.out(self.act(self.drop(self.fc1(x))))


class PreBDI(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_classes, model_args=None):
        super(PreBDI, self).__init__()

        # attributes
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # model layer
        self.encoder_layer = celk(model_args['cross_data_dim'], model_args['cross_in_len'], model_args['cross_out_len'],
                                     model_args['cross_seg_len'], model_args['cross_win_size'], model_args['cross_factor'],
                                     model_args['cross_d_model'], model_args['cross_d_ff'], model_args['cross_n_heads'],
                                     model_args['cross_e_layers'], model_args['cross_dropout'], model_args['cross_baseline'],
                                     mode='finetune')

        self.preConv = nn.Sequential()
        self.preConv.append(nn.Conv1d(in_channels=5632, out_channels=128, kernel_size=self.num_nodes, padding='same'))
        self.preConv.append(nn.BatchNorm1d(128))
        self.preConv.append(nn.LeakyReLU())
        self.ConvLSTM = ConvLSTMBlock(channels=1, hidden_dim=[32], nums_layer=1)
        self.ConvBN = nn.BatchNorm3d(num_features=128)
        self.Flatten = nn.Flatten(2)
        self.att = Attention([None, 128, 32 * self.num_nodes])

        self.out = SimpleOut(in_dim=32 * self.num_nodes, out_dim=self.output_len, hidden_dim=512)

    def forward(self, X, padding_masks):
        batch_size, seq_len, num_nodes, _ = X.shape

        time_series = X[..., 0]
        temperature = X[..., 1]
        time_series_emb = self.encoder_layer(time_series, temperature)
        hidden = time_series_emb

        hidden = self.preConv(hidden)
        # 这里根据不同的数据集做修改，如简支梁桥20个传感器可分为两层(2, 10), 钢框架16个传感器分为4层(4,4)
        hidden = self.ConvLSTM(hidden, 2, 7)
        hidden = self.ConvBN(hidden)
        hidden = self.Flatten(hidden)
        hidden = self.att(hidden)
        hidden = hidden[0]

        output = self.out(hidden.contiguous().view(batch_size, -1))

        return output
class ACLBDI(nn.Module):
    def __init__(self, num_classes = 9, hidden_size = 512, if_domain = False):
        super(ACLBDI, self).__init__()

        self.num_classes = num_classes
        self.if_domain = if_domain
        self.hidden_size = hidden_size

        self.ConvLSTM = ConvLSTMBlock(channels=1, hidden_dim=[32], nums_layer=1)
        self.ConvBN = nn.BatchNorm3d(num_features=128)
        self.flatten = nn.Flatten(2)
        self.att = Attention([None, 128, 512])
        self.ACL_output_layer = SimpleOut(in_dim=512, out_dim=9, hidden_dim=512)
        # 分类任务
        # self.out = nn.Softmax(dim=1)

        if self.if_domain:
            self.flatten = nn.Flatten()
            self.build_classifier()
            self.build_discriminator()

    def build_classifier(self):
        self.classifier = nn.Sequential()
        # self.classifier.append(Attention([None, 1280, self.num_nodes * self.num_nodes]))
        self.classifier.append(nn.Linear(in_features=65536, out_features=256))
        self.classifier.append(nn.BatchNorm1d(256))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Linear(in_features=256, out_features=128))
        self.classifier.append(nn.BatchNorm1d(128))
        # self.classifier.append(nn.Dropout(0.5))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Linear(in_features=128, out_features=9))
        self.classifier.append(nn.BatchNorm1d(9))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Softmax(dim=1))
    def build_discriminator(self):
        self.discriminator = nn.Sequential()
        self.discriminator.append(GradientReverseLayer())
        self.discriminator.append(nn.Linear(in_features=65536, out_features=128))
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


    def forward(self, X, mask=None):
        time_series = X[..., 0]
        hidden = self.ConvLSTM(time_series, 4, 4)
        hidden = self.ConvBN(hidden)
        hidden = self.flatten(hidden)
        hidden = hidden.view(hidden.shape[0], hidden.shape[1], -1)
        hidden = self.att(hidden)
        hidden = hidden[0]
        output = self.ACL_output_layer(hidden)

        # 分类任务(不做域适应)
        # output = self.out(output)

        if self.if_domain:
            output = self.classifier(hidden)
            domain = self.discriminator(hidden)
            return output, domain

        return output

class PreDBDI(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_classes, model_args=None):
        super(PreDBDI, self).__init__()

        # attributes
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.if_domain = model_args["if_domain"]
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.encoder_layer = Crossformer(model_args['cross_data_dim'], model_args['cross_in_len'], model_args['cross_out_len'], model_args['cross_seg_len'],
                               model_args['cross_win_size'], model_args['cross_factor'], model_args['cross_d_model'], model_args['cross_d_ff'],
                               model_args['cross_n_heads'], model_args['cross_e_layers'], model_args['cross_dropout'], model_args['cross_baseline'], mode='finetune')

        self.preConv = nn.Sequential()
        self.preConv.append(nn.Conv1d(in_channels=5632, out_channels=128, kernel_size=self.num_nodes, padding='same'))
        self.preConv.append(nn.BatchNorm1d(128))
        self.preConv.append(nn.LeakyReLU())

        self.ConvLSTM = ConvLSTMBlock(channels=1, hidden_dim=[self.num_nodes], kernal_size=(2,2), nums_layer=1)
        self.ConvBN = nn.BatchNorm3d(num_features=128)

        if self.if_domain:
            self.Flatten = nn.Flatten()
            self.build_classifier()
            self.build_discriminator()
        else:
            self.Flatten = nn.Flatten(2)
            self.att = Attention([None, 128, self.num_nodes * self.num_nodes])
            self.sout = SimpleOut(in_dim=self.num_nodes * self.num_nodes, out_dim=self.output_len, hidden_dim=512)
            self.out = nn.Softmax(dim=1)

    def build_classifier(self):
        self.classifier = nn.Sequential()
        # self.classifier.append(Attention([None, 1280, self.num_nodes * self.num_nodes]))
        self.classifier.append(nn.Linear(in_features=self.num_nodes * self.num_nodes * self.input_len, out_features=256))
        self.classifier.append(nn.BatchNorm1d(256))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Linear(in_features=256, out_features=128))
        self.classifier.append(nn.BatchNorm1d(128))
        # self.classifier.append(nn.Dropout(0.5))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Linear(in_features=128, out_features=self.output_len))
        self.classifier.append(nn.BatchNorm1d(self.output_len))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Softmax(dim=1))

    def build_discriminator(self):
        self.discriminator = nn.Sequential()
        self.discriminator.append(GradientReverseLayer())
        self.discriminator.append(nn.Linear(in_features=self.num_nodes * self.num_nodes *self.input_len, out_features=128))
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

    def forward(self, X, padding_masks):

        B,L,N,_ = X.shape
        time_series = X[..., 0]
        time_series_emb = self.encoder_layer(time_series)
        hidden = self.STIDSimple(X, time_series_emb)

        hidden = self.preConv(hidden)
        hidden = self.ConvLSTM(hidden, 4, 4)
        hidden = self.ConvBN(hidden)
        hidden = self.Flatten(hidden)


        if self.if_domain:
            label_out = self.classifier(hidden)
            domain_out = self.discriminator(hidden)
            return label_out, domain_out
        else:
            hidden = self.att(hidden)
            hidden = hidden[0]
            output = self.sout(hidden.contiguous().view(B, -1))
            output = self.out(output)
            return output




